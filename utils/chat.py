"""
chat.py
负责：
1. 构建 RAG（检索增强生成）的 prompt
2. 调用 Claude API 生成回答
3. 管理多轮对话历史
"""

import os
import anthropic
from typing import List, Dict, Tuple


# 多轮对话最多保留的「轮数」（1 轮 = 1 条 user + 1 条 assistant）。
# 超出后丢弃最早的轮次，防止历史无限增长导致 token 膨胀、变慢变贵。
MAX_HISTORY_TURNS: int = 6


def get_anthropic_client() -> anthropic.Anthropic:
    """
    初始化 Anthropic 客户端。
    API Key 从环境变量 ANTHROPIC_API_KEY 读取，不硬编码在代码里。

    Returns:
        anthropic.Anthropic 客户端实例

    Raises:
        ValueError: 若环境变量未设置
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 ANTHROPIC_API_KEY 环境变量，"
            "请在终端执行: export ANTHROPIC_API_KEY='your-key-here'"
        )
    return anthropic.Anthropic(api_key=api_key)


def build_system_prompt() -> str:
    """
    构建系统级 prompt，告诉 Claude 它的角色和行为规范。

    Returns:
        系统 prompt 字符串
    """
    return """你是一位专业的行业研究助手，擅长分析研究报告、白皮书和行业文档。

你的工作方式：
1. 【严格基于参考资料】只能使用下方【参考资料】中的信息作答，不得依赖你自己的先验知识，也不得推测或脑补
2. 回答时引用具体来源，格式为「来源：《文件名》第X段」
3. 【资料不足时如实说明】若参考资料中没有足够信息回答问题，必须明确回答「根据现有资料，我无法回答这个问题」，并指出还缺少什么信息——绝不用通用知识填补、绝不编造
4. 回答结构清晰，使用条目、小标题等格式提升可读性
5. 使用中文回答（除非用户用其他语言提问）
6. 保持客观、专业，不夸大或歪曲资料内容"""


def build_rag_context(retrieved_chunks: List[Dict]) -> str:
    """
    将检索到的文本块格式化成 Claude 可理解的参考资料块。

    Args:
        retrieved_chunks: embedder.query_similar_chunks() 返回的结果列表

    Returns:
        格式化后的参考资料字符串，直接嵌入 user message
    """
    if not retrieved_chunks:
        return "【当前知识库中未找到相关内容】"

    context_parts = ["【参考资料】（按相关度排序）\n"]
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        # 相关度：余弦距离越小越好，转换成百分比展示
        relevance = round((1 - chunk["distance"]) * 100, 1)
        context_parts.append(
            f"---\n"
            f"[资料{i}] 来源：《{source}》第{chunk_id + 1}段 "
            f"（相关度：{relevance}%）\n"
            f"{text}\n"
        )

    return "\n".join(context_parts)


def build_user_message(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    将用户问题和检索到的参考资料拼接成完整的 user message。

    Args:
        question:         用户的原始问题
        retrieved_chunks: 检索结果列表

    Returns:
        拼接好的 user message 字符串
    """
    rag_context = build_rag_context(retrieved_chunks)
    return f"{rag_context}\n\n【用户问题】\n{question}"


def rewrite_query(
    question: str,
    conversation_history: List[Dict],
    model_name: str,
) -> str:
    """
    多轮检索改写：把可能含指代/省略的追问，结合对话历史改写成
    一个独立、完整、适合向量检索的查询。

    为什么需要：用户第二句问「那它的毛利率呢？」，"它"指上文的某公司。
    若直接拿这句去检索，检索系统根本不知道"它"是谁 → 检索全废。
    改写成「台积电 毛利率」再检索，才能命中。

    设计要点：
    - 首轮（无历史）直接返回原问题，省一次 API 调用
    - 任何异常都安全回退到原问题，绝不阻断主流程（比如没配 API Key 时）

    Args:
        question:             用户当前问题
        conversation_history: Claude messages 格式的历史
        model_name:           用于改写的模型（可换成更便宜更快的 Haiku）

    Returns:
        改写后的查询字符串；首轮或失败时原样返回 question
    """
    if not conversation_history:
        return question  # 首轮无上下文，无需改写

    try:
        client = get_anthropic_client()
        # 只取最近几轮，控制 token
        recent = list(conversation_history)[-(MAX_HISTORY_TURNS * 2):]
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
            for m in recent
        )
        prompt = (
            "下面是一段对话历史和用户的最新问题。\n"
            "请把【最新问题】改写成一个独立、完整、适合知识库检索的查询：\n"
            "- 把代词（它、这个、那家公司等）替换成具体所指\n"
            "- 补全省略的主语和背景\n"
            "- 只输出改写后的查询本身，不要任何解释、前缀或引号\n\n"
            f"【对话历史】\n{history_text}\n\n"
            f"【最新问题】\n{question}\n\n"
            "改写后的查询："
        )
        resp = client.messages.create(
            model=model_name,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = resp.content[0].text.strip()
        return rewritten or question
    except Exception:
        # 任何出错（无 Key、网络等）都回退原问题，保证检索不中断
        return question


def chat_with_claude(
    question: str,
    retrieved_chunks: List[Dict],
    conversation_history: List[Dict],
    model_name: str,
    max_tokens: int = 2048,
) -> Tuple[str, List[Dict]]:
    """
    调用 Claude API 进行 RAG 问答，支持多轮对话。

    Args:
        question:              用户当前输入的问题
        retrieved_chunks:      检索到的相关文本块（用于构建上下文）
        conversation_history:  之前的对话历史，格式为 Claude API 的 messages 列表
        model_name:            Claude 模型 ID（如 claude-sonnet-4-20250514）
        max_tokens:            生成回答的最大 token 数

    Returns:
        Tuple:
            - answer (str): Claude 生成的回答文本
            - updated_history (List[Dict]): 加入本轮对话后的完整历史
    """
    client = get_anthropic_client()

    # 构建本轮 user message（包含检索上下文 + 用户问题）
    user_message_content = build_user_message(question, retrieved_chunks)

    # 滑动窗口：只把最近 MAX_HISTORY_TURNS 轮历史喂给 API，
    # 防止长对话 token 无限累积。截取以 2 条/轮 计算，从尾部保留。
    recent_history = list(conversation_history)[-(MAX_HISTORY_TURNS * 2):]

    # 将本轮消息追加到（裁剪后的）历史（不修改原始列表）
    messages = recent_history + [
        {"role": "user", "content": user_message_content}
    ]

    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        system=build_system_prompt(),
        messages=messages,
    )

    answer = response.content[0].text

    # 更新对话历史（保存原始问题，不含检索上下文，保持历史简洁）
    updated_history = list(conversation_history) + [
        {"role": "user", "content": question},          # 历史中只存用户原始问题
        {"role": "assistant", "content": answer},
    ]

    return answer, updated_history


def format_sources_for_display(retrieved_chunks: List[Dict]) -> str:
    """
    将检索到的来源信息格式化成用户友好的 markdown 字符串，
    在界面上展示在 Claude 回答的下方。

    Args:
        retrieved_chunks: 检索结果列表

    Returns:
        markdown 格式的来源摘要字符串
    """
    if not retrieved_chunks:
        return ""

    lines = ["---", "**📚 本次检索参考来源：**"]
    seen = set()
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        relevance = round((1 - chunk["distance"]) * 100, 1)
        key = f"{source}_{chunk_id}"
        if key not in seen:
            seen.add(key)
            # 截取预览文本（前80字）
            preview = chunk["text"][:80].replace("\n", " ")
            lines.append(
                f"{i}. **《{source}》** 第{chunk_id + 1}段 "
                f"（相关度 {relevance}%）  \n"
                f"   *预览：{preview}…*"
            )

    return "\n".join(lines)
