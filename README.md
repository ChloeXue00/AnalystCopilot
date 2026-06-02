# AnalystCopilot

> 把一摞研报变成可问答的知识库——几秒钟定位答案，让案头研究从「读一整天」变成「问一句话」。

一个面向行业研究 / 投研场景的 RAG 问答助手：上传 PDF 研报，用自然语言提问，由 Claude 基于报告内容回答并**标注来源**，拒绝凭空编造。

### 🚀 最新版本已实测上线，欢迎点击下方链接使用

**👉 [analystcopilot-turbo.streamlit.app](https://analystcopilot-turbo.streamlit.app/)** 

---

## 它解决什么问题

分析师做案头研究（desk research）时，最耗时的不是思考，而是**在几十份 PDF 里翻找一个数字、一句结论**。AnalystCopilot 把这件事自动化：

- **找得快**：语义检索，问"芯片产业前景"也能命中写着"半导体行业"的段落，不必猜关键词
- **答得准**：回答严格基于上传的资料，并标注「来源：《某报告》第 X 段」
- **不瞎编**：资料里没有的，直接回答"根据现有资料无法回答"，而不是用通用知识糊弄

---

## 技术架构

```
PDF 上传
   ↓
pdfplumber 解析：正文 / 表格分离，表格转 Markdown          (utils/pdf_parser.py)
   ↓
按段落切分（~400 字，50 字重叠）；表格整块保留
   ↓
Voyage API 向量化（voyage-3，云端调用，本地零模型依赖）      (utils/embedder.py)
   ↓
纯 numpy 本地向量库（pickle 持久化，替代 chromadb）
   ↓
用户提问 → [可选] LLM 改写 → 向量检索粗筛 + 距离阈值过滤
   ↓
[可选] Voyage rerank API 重排，精选 top-k                     (rerank)
   ↓
Claude API 基于检索内容生成回答，严格 grounding              (utils/chat.py)
   ↓
返回答案 + 来源标注
```

## 文件结构

```
AnalystCopilot/
├── app.py                  # Streamlit 主程序，顶部集中所有可调参数
├── utils/
│   ├── pdf_parser.py       # PDF 解析、表格感知切分、表格质量过滤
│   ├── embedder.py         # Voyage 向量化/重排 + 纯 numpy 向量库
│   ├── chat.py             # Claude API 调用、严格提示词、query rewriting
│   └── rate_limit.py       # 全站每日提问限流（保护 API key）
├── eval_retrieval.py       # 检索质量评估（recall@k / precision@k）
├── requirements.txt
├── start_app.bat           # Windows 一键启动
└── README.md
```

---

## 快速开始

```bash
# 1. 安装依赖（纯 API 方案，无 torch，依赖很轻）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置两个 API Key（向量检索用 Voyage，生成回答用 Claude）
#   macOS / Linux:
export ANTHROPIC_API_KEY="sk-ant-xxxxxx"
export VOYAGE_API_KEY="pa-xxxxxx"
#   Windows PowerShell:
$env:ANTHROPIC_API_KEY="sk-ant-xxxxxx"
$env:VOYAGE_API_KEY="pa-xxxxxx"

# 3. 启动
streamlit run app.py
```

也可复制 `.streamlit/secrets.toml.example` 为 `.streamlit/secrets.toml` 填入 Key。浏览器打开 `http://localhost:8501`。

---

## 可调参数（`app.py` 顶部）

```python
CHUNK_SIZE    = 400     # 每块目标字符数
OVERLAP       = 50      # 相邻块重叠字符数
TOP_K         = 5       # 最终给 LLM 的段落数（精排后）
RETRIEVE_K    = 20      # 向量粗筛候选数（保召回）
MAX_DISTANCE  = 0.8     # 相关度阈值，超过则视为不相关丢弃
ENABLE_RERANK = False   # 是否启用 cross-encoder 精排（需先下载 reranker）
```

用 `python eval_retrieval.py` 可以量化这些参数的效果。

---

## 技术栈

Streamlit · Claude API · Voyage AI（embedding + rerank）· pdfplumber · numpy

## 已知限制

- 复杂多层表头 / 无边框表格的解析仍不完美（pdfplumber 局限）
- 云端为临时文件存储：每个会话独立知识库，应用休眠/重启后上传内容会清空（适合 demo，非长期资料库）
- "全文共有几个表格"这类需纵览全文的问题，RAG 天然不擅长（适合"关于 X 的内容是什么"）
