"""
app.py
行研知识库助手 - 主程序入口

架构：
- 左侧侧边栏：PDF 上传、文件管理
- 主区域：多轮对话界面
- 后端：ChromaDB 向量检索 + Claude API 生成回答
"""

import os
import tempfile
import streamlit as st
from pathlib import Path

from utils.pdf_parser import parse_pdf
from utils.embedder import (
    load_embedding_model,
    load_reranker,
    rerank_chunks,
    add_chunks_to_db,
    query_similar_chunks,
    delete_source_from_db,
    get_all_sources,
    get_chunk_count_per_source,
)
from utils.chat import chat_with_claude, format_sources_for_display, rewrite_query
from utils.rate_limit import check_and_increment, get_count

# ============================================================
# 全局配置常量（在此修改以调整系统行为）
# ============================================================

CHUNK_SIZE: int = 400          # 每个文本块的目标字符数（300-500 推荐）
OVERLAP: int = 50              # 相邻文本块的重叠字符数
TOP_K: int = 5                 # 【最终】给 LLM 的段落数（第二阶段精排后保留，保 precision）
RETRIEVE_K: int = 20           # 【粗筛】向量检索的候选数（第一阶段撒大网，保 recall）
MAX_DISTANCE: float = 0.8      # 相关度阈值：余弦距离>此值的块视为不相关被丢弃（≈相关度<20%），
                               # 防止"库里没相关内容时硬塞垃圾→幻觉"。用 eval_retrieval.py 调优此值
ENABLE_RERANK: bool = False    # 第二阶段精排开关（Voyage rerank API）。
                               # 开启会对每次查询多调一次 rerank API，按需开启。
RERANK_MODEL: str = "rerank-2"                  # Voyage 重排模型
EMBEDDING_MODEL: str = "voyage-3"               # Voyage 嵌入模型（金融可换 voyage-finance-2）
CLAUDE_MODEL: str = "claude-sonnet-4-20250514"                   # Claude 模型 ID
MAX_TOKENS: int = 2048         # Claude 生成回答的最大 token 数
CHROMA_COLLECTION: str = "rag_docs"  # 向量库集合名称

DAILY_LIMIT: int = 50          # 【演示限流】用作者自带 key 时，全站每日总提问上限。
                               # 访客在侧边栏填入自己的 key 后不受此限。跨天自动归零。

# ============================================================
# 页面基础配置
# ============================================================

st.set_page_config(
    page_title="行研知识库助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# API Key 加载（云端部署用作者自己的 key）
# ============================================================

def load_api_keys_from_secrets():
    """
    从 Streamlit secrets 读取 API key 写入环境变量。
    部署到 Streamlit Cloud 时，在 App → Settings → Secrets 里配置：
        ANTHROPIC_API_KEY = "sk-ant-..."
        VOYAGE_API_KEY    = "pa-..."
    本地无 secrets.toml 时 st.secrets 访问会抛错，已 try/except 忽略，
    此时回退到「访客自带 key」模式（侧边栏手填）。
    """
    try:
        for key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
            value = st.secrets.get(key)
            if value and not os.environ.get(key):
                os.environ[key] = str(value)
    except Exception:
        pass


# ============================================================
# 全局资源缓存（避免每次刷新重新加载模型）
# ============================================================

@st.cache_resource(show_spinner="正在初始化 embedding 客户端...")
def get_embedding_model():
    """
    缓存加载 SentenceTransformer 模型。
    @st.cache_resource 保证整个 session 只加载一次。
    """
    return load_embedding_model(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="正在加载重排模型...")
def get_reranker():
    """
    缓存加载 cross-encoder 重排模型。
    若未启用或加载失败（如模型没下载），返回 None —— 系统自动退回纯向量检索，
    保证 app 在没有 reranker 时也能正常跑。
    """
    if not ENABLE_RERANK:
        return None
    try:
        return load_reranker(RERANK_MODEL)
    except Exception as e:
        st.warning(f"⚠️ 重排模型加载失败，已退回纯向量检索：{e}")
        return None


# ============================================================
# Session State 初始化（保持对话状态跨 rerun 存活）
# ============================================================

def init_session_state():
    """
    初始化 Streamlit session state 中的关键变量。
    session_state 在同一浏览器会话内跨 rerun 持久存在。
    """
    if "conversation_history" not in st.session_state:
        # Claude API messages 格式的对话历史
        st.session_state.conversation_history = []

    if "display_messages" not in st.session_state:
        # 用于 UI 展示的消息列表（含来源信息）
        st.session_state.display_messages = []

    if "uploaded_files_set" not in st.session_state:
        # 已上传文件名集合（用于避免重复上传）
        st.session_state.uploaded_files_set = set(get_all_sources(CHROMA_COLLECTION))


# ============================================================
# 侧边栏：文件上传与管理
# ============================================================

def render_sidebar(model):
    """
    渲染左侧侧边栏，包含：
    1. PDF 批量上传区域
    2. 已上传文件列表（带删除按钮）
    3. 系统状态信息

    Args:
        model: 已加载的 SentenceTransformer 模型实例
    """
    with st.sidebar:
        st.title("📊 行研知识库助手")
        st.markdown("---")

        # ── 0. API Key 配置 ──────────────────────────────
        st.subheader("🔑 API 配置")

        # 是否已由作者的 secrets 提供了可用的演示额度
        demo_ready = bool(os.environ.get("ANTHROPIC_API_KEY")) and bool(
            os.environ.get("VOYAGE_API_KEY")
        )

        if demo_ready and not st.session_state.get("byo_key"):
            used = get_count()
            remaining = max(DAILY_LIMIT - used, 0)
            st.success(f"✅ 已启用免费演示额度（每日 {DAILY_LIMIT} 次提问）")
            st.caption(
                f"今日剩余 **{remaining}/{DAILY_LIMIT}** 次。"
                "额度用完后，可在下方填入自己的 API Key 不限次体验。"
            )

        # 访客自带 key（可选）：填入后绕过演示额度限制，费用走访客自己的账户
        with st.expander(
            "🔧 用自己的 API Key（可选，不限次数）",
            expanded=not demo_ready,  # 没有演示额度时默认展开，引导填写
        ):
            api_key_input = st.text_input(
                "Anthropic API Key",
                type="password",
                help="从 https://console.anthropic.com/ 获取（用于生成回答）",
            )
            voyage_key_input = st.text_input(
                "Voyage API Key",
                type="password",
                help="从 https://www.voyageai.com/ 获取（用于文本向量化/检索）",
            )
            if api_key_input and voyage_key_input:
                os.environ["ANTHROPIC_API_KEY"] = api_key_input
                os.environ["VOYAGE_API_KEY"] = voyage_key_input
                st.session_state.byo_key = True
                st.success("已切换为你自己的 API Key，不受演示额度限制。")
            elif api_key_input or voyage_key_input:
                st.warning("两个 Key 都需要填写才能生效。")

        st.markdown("---")

        # ── 1. 文件上传区域 ──────────────────────────────
        st.subheader("📁 上传研究报告")
        uploaded_files = st.file_uploader(
            label="支持批量上传 PDF 文件",
            type=["pdf"],
            accept_multiple_files=True,
            help="上传后系统自动解析文本并建立向量索引",
        )

        if uploaded_files:
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.uploaded_files_set
            ]

            if new_files:
                _process_uploaded_files(new_files, model)
            else:
                st.info("所选文件均已上传，无需重复处理。")

        st.markdown("---")

        # ── 2. 已上传文件列表 ────────────────────────────
        st.subheader("📚 知识库文件")
        chunk_counts = get_chunk_count_per_source(CHROMA_COLLECTION)

        if not chunk_counts:
            st.caption("知识库为空，请先上传 PDF 文件。")
        else:
            for source_name, count in chunk_counts.items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    # 文件名过长时截断显示
                    display_name = source_name if len(source_name) <= 22 else source_name[:19] + "..."
                    st.markdown(f"📄 **{display_name}**  \n`{count} 个片段`")
                with col2:
                    if st.button("🗑️", key=f"del_{source_name}", help=f"删除 {source_name}"):
                        _delete_file(source_name)
                        st.rerun()

        st.markdown("---")

        # ── 3. 对话控制 ──────────────────────────────────
        st.subheader("⚙️ 对话控制")
        if st.button("🧹 清空对话历史", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.display_messages = []
            st.rerun()

        # ── 4. 系统状态信息 ──────────────────────────────
        with st.expander("ℹ️ 系统配置", expanded=False):
            st.code(
                f"嵌入模型: {EMBEDDING_MODEL}\n"
                f"重排模型: {RERANK_MODEL if ENABLE_RERANK else '（未启用）'}\n"
                f"对话模型: {CLAUDE_MODEL}\n"
                f"块大小: {CHUNK_SIZE} 字符\n"
                f"重叠大小: {OVERLAP} 字符\n"
                f"粗筛候选: Top-{RETRIEVE_K} → 精排保留: Top-{TOP_K}\n"
                f"相关度阈值: 距离≤{MAX_DISTANCE}\n"
                f"知识库路径: ./vector_store",
                language="yaml",
            )


def _process_uploaded_files(new_files, model):
    """
    处理新上传的 PDF 文件：解析 → 向量化 → 存入 ChromaDB。

    Args:
        new_files: 尚未入库的 UploadedFile 对象列表
        model:     SentenceTransformer 模型实例
    """
    progress_bar = st.progress(0, text="准备处理文件...")

    for i, uploaded_file in enumerate(new_files):
        progress_text = f"正在处理 ({i + 1}/{len(new_files)}): {uploaded_file.name}"
        progress_bar.progress((i) / len(new_files), text=progress_text)

        # 将上传的文件临时保存到磁盘（pdfplumber 需要文件路径）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # 解析 PDF 并切块
            chunks = parse_pdf(tmp_path, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

            # 修正 source 字段为原始文件名（临时文件名无意义）
            for chunk in chunks:
                chunk["source"] = uploaded_file.name

            # 向量化并写入数据库
            count = add_chunks_to_db(chunks, model, collection_name=CHROMA_COLLECTION)

            st.session_state.uploaded_files_set.add(uploaded_file.name)
            st.sidebar.success(f"✅ {uploaded_file.name}（{count} 段）")

        except Exception as e:
            st.sidebar.error(f"❌ {uploaded_file.name} 处理失败：{e}")

        finally:
            # 清理临时文件
            Path(tmp_path).unlink(missing_ok=True)

    progress_bar.progress(1.0, text="所有文件处理完成！")


def _delete_file(source_name: str):
    """
    从 ChromaDB 删除指定文件的所有向量数据，并更新 session state。

    Args:
        source_name: 要删除的文件名
    """
    remaining = delete_source_from_db(source_name, collection_name=CHROMA_COLLECTION)
    st.session_state.uploaded_files_set.discard(source_name)
    st.toast(f"已删除《{source_name}》，库中剩余 {remaining} 条记录。")


# ============================================================
# 主区域：对话界面
# ============================================================

def render_chat_area(model, reranker):
    """
    渲染主对话区域：
    1. 展示历史对话气泡
    2. 用户输入框
    3. 触发检索 + Claude API 调用
    4. 流式展示回答并显示来源

    Args:
        model:    已加载的 SentenceTransformer 模型实例
        reranker: cross-encoder 重排模型（可能为 None，表示不重排）
    """
    st.title("💬 行研问答")

    # 检查知识库是否有内容
    sources = get_all_sources(CHROMA_COLLECTION)
    if not sources:
        st.info("👈 请先在左侧上传 PDF 研究报告，然后开始提问。")
    else:
        st.caption(f"知识库已加载 {len(sources)} 个文件，可以开始提问了。")

    # ── 渲染历史对话气泡 ──────────────────────────────────
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 展示来源信息（仅 assistant 消息有）
            if "sources" in msg and msg["sources"]:
                with st.expander("查看参考来源", expanded=False):
                    st.markdown(msg["sources"])

    # ── 用户输入框 ────────────────────────────────────────
    user_input = st.chat_input(
        placeholder="请输入您的问题，例如：这份报告的核心结论是什么？",
        disabled=(len(sources) == 0),
    )

    if user_input:
        _handle_user_input(user_input, model, reranker)


def _handle_user_input(question: str, model, reranker):
    """
    处理用户输入的完整流程：
    1. 显示用户消息气泡
    2. 检索相关文本块（两阶段：向量粗筛 → cross-encoder 精排）
    3. 调用 Claude API 获取回答
    4. 显示回答和来源
    5. 更新对话历史

    Args:
        question: 用户输入的问题字符串
        model:    SentenceTransformer 模型实例
        reranker: cross-encoder 重排模型（None 则跳过精排）
    """
    # ── 演示限流：用作者自带 key 时，全站每日总提问数有上限 ──────────
    # 访客填了自己的 key（byo_key=True）则不受限，费用走他自己账户。
    if not st.session_state.get("byo_key", False):
        allowed, _ = check_and_increment(DAILY_LIMIT)
        if not allowed:
            with st.chat_message("assistant"):
                st.warning(
                    f"😅 今日免费演示额度（{DAILY_LIMIT} 次）已用完。\n\n"
                    "请明天再来，或在左侧 **🔧 用自己的 API Key** 填入你的 "
                    "Anthropic + Voyage Key，即可不限次数继续体验。"
                )
            return

    # 立即显示用户消息
    with st.chat_message("user"):
        st.markdown(question)

    # 追加到展示历史
    st.session_state.display_messages.append({
        "role": "user",
        "content": question,
    })

    # ── 多轮改写 + 检索相关段落 ───────────────────────────
    with st.spinner("🔍 正在检索知识库..."):
        # 把含指代/省略的追问（"它呢？"）改写成独立查询，再去检索
        search_query = rewrite_query(
            question=question,
            conversation_history=st.session_state.conversation_history,
            model_name=CLAUDE_MODEL,
        )
        # 第一阶段（粗筛）：向量检索撒大网取 RETRIEVE_K 个候选，保 recall
        # 检索仍用改写后的查询；展示和生成仍用用户原始问题
        candidates = query_similar_chunks(
            query=search_query,
            model=model,
            top_k=RETRIEVE_K,
            collection_name=CHROMA_COLLECTION,
            max_distance=MAX_DISTANCE,
        )

        # 第二阶段（精排）：cross-encoder 重排，取 top TOP_K，保 precision
        # reranker 为 None（未启用/加载失败）时，直接退回向量排序的前 TOP_K
        if reranker is not None and candidates:
            retrieved_chunks = rerank_chunks(search_query, candidates, reranker, top_n=TOP_K)
        else:
            retrieved_chunks = candidates[:TOP_K]

    # 改写生效时给用户一个透明提示（看到 rewriting 真的在工作）
    if search_query != question:
        st.caption(f"🔁 检索查询已改写为：{search_query}")

    # ── 调用 Claude API 生成回答 ──────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("🤔 Claude 正在分析和生成回答..."):
            try:
                answer, updated_history = chat_with_claude(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                    conversation_history=st.session_state.conversation_history,
                    model_name=CLAUDE_MODEL,
                    max_tokens=MAX_TOKENS,
                )
            except ValueError as e:
                # API Key 未配置
                st.error(f"⚠️ 配置错误：{e}")
                return
            except Exception as e:
                st.error(f"⚠️ 调用 Claude API 时出错：{e}")
                return

        # 展示回答
        st.markdown(answer)

        # 展示来源
        sources_text = format_sources_for_display(retrieved_chunks)
        if sources_text:
            with st.expander("查看参考来源", expanded=False):
                st.markdown(sources_text)

    # ── 更新 session state ────────────────────────────────
    st.session_state.conversation_history = updated_history
    st.session_state.display_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_text,
    })


# ============================================================
# 主程序入口
# ============================================================

def main():
    """
    应用主入口：初始化状态 → 加载模型 → 渲染 UI。
    """
    # 优先从 Streamlit secrets 载入作者的 key（云端演示额度）
    load_api_keys_from_secrets()

    init_session_state()

    # 初始化 embedding / rerank 客户端（轻量对象，真正的 API 调用在用到时才发生，
    # 此时侧边栏已把 API Key 写入环境变量）
    model = get_embedding_model()
    reranker = get_reranker()  # 未启用时为 None → 自动退回纯向量检索

    # 先渲染侧边栏：让用户填入 API Key（在任何 API 调用之前）
    render_sidebar(model)

    # 再渲染主对话区域
    render_chat_area(model, reranker)


if __name__ == "__main__":
    main()
