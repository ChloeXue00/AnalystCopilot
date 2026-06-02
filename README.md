# AnalystCopilot

> 把一摞研报变成可问答的知识库——几秒钟定位答案，让案头研究从「读一整天」变成「问一句话」。

一个面向行业研究 / 投研场景的 RAG 问答助手：上传 PDF 研报，用自然语言提问，由 Claude 基于报告内容回答并**标注来源**，拒绝凭空编造。

### 🚀 在线体验

**👉 [analystcopilot-turbo.streamlit.app](https://analystcopilot-turbo.streamlit.app/)** — 无需安装，打开即用，上传你自己的 PDF 研报试试。

> 部署在 Streamlit Community Cloud（免费版），空闲一段时间会休眠，首次打开如显示唤醒页，点一下按钮等约 1 分钟即可。演示额度由作者提供（每日全站 50 次提问），请文明使用 🙏。

---

## 它解决什么问题

分析师做案头研究（desk research）时，最耗时的不是思考，而是**在几十份 PDF 里翻找一个数字、一句结论**。AnalystCopilot 把这件事自动化：

- **找得快**：语义检索，问"芯片产业前景"也能命中写着"半导体行业"的段落，不必猜关键词
- **答得准**：回答严格基于上传的资料，并标注「来源：《某报告》第 X 段」
- **不瞎编**：资料里没有的，直接回答"根据现有资料无法回答"，而不是用通用知识糊弄

---

## 这个项目里做了哪些「产品决策」

> RAG 搭起来不难，难的是发现并解决真实场景里的缺陷。以下每一条都对应一个被实测验证过的问题。

| 发现的问题 | 根因 | 采取的方案 | 效果 |
|-----------|------|-----------|------|
| 研报里的**财务表格丢失** | `extract_text` 把表格压平成一团数字，行列对应关系全断 | 表格单独抽取 → 转 Markdown 整块保留，不参与字符切割 | 表格变成可检索的结构化单元 |
| 表格里混入大量**图表噪声** | pdfplumber 把柱状图/折线图误判成表格，抽出一堆空格和坐标碎片 | 加表格**质量过滤**（非空率、有效字符数启发式） | 真实报告中过滤掉 **75%** 的垃圾表格块（44→11） |
| 多轮对话**指代检索失效** | 追问"那它的毛利率呢"，检索系统不知道"它"指谁 | 检索前用 LLM 做 **query rewriting**，补全指代 | "它"→具体主体，检索命中率提升 |
| 库里没相关内容时**硬塞垃圾→幻觉** | 无脑返回 top-k，哪怕全不相关 | 加**相关度阈值**，距离过大的块直接丢弃 | 无关问题触发"无法回答"而非编造 |
| 单阶段向量检索**精度有限** | 向量检索快但粗 | **两阶段检索**：向量粗筛 top-20 → rerank 精排 top-5 | 召回与精度兼得（可选开关） |
| 改了参数**不知道是否变好** | 全凭感觉调 chunk_size / top_k | 写了 **recall@k / precision@k 评估脚本** | 任何优化都能用数字验证 |

> **为了"能上线"做的两个关键取舍**：
> 1. `chromadb` 在本机原生模块崩溃（segfault）→ **用纯 numpy 自实现轻量向量库**替代，零原生依赖、永不崩溃、逻辑透明（见 `utils/embedder.py`）。
> 2. 本地 `sentence-transformers`（torch）在 Streamlit Cloud 免费版 1GB 内存下 **OOM 崩溃** → 改用 **Voyage embedding/rerank API**，本地内存占用极小，让应用真正跑得上免费云环境。这是"本地能跑"到"线上可交互"的决定性一步。

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

## 云端部署 & 成本治理

把一个"本地能跑的脚本"变成"任何人点链接就能用的公开产品"，需要解决一批本地不会遇到的问题：

| 上线挑战 | 解决方案 |
|---------|---------|
| **公开链接 = 谁都能花我的钱** | 全站**每日提问限流**（`utils/rate_limit.py`，跨天自动归零）+ Anthropic/Voyage 后台预算控制，成本可控可预测 |
| **多人同时用，互相看到彼此的文件** | 每个浏览器**会话独立向量库**（`rag_<uuid>`），上传互不可见；并定期清理过期会话文件 |
| **API Key 不能暴露在前端** | Key 走 **Streamlit secrets** 注入，前端完全不展示，访客无需配置、开箱即用 |
| **免费版 1GB 内存装不下本地大模型** | 向量化/重排改为 **API 调用**，本地零模型依赖（见上文取舍） |
| **`headless=false` 导致云端启动崩溃** | 服务器环境强制 `headless=true`，避免首次运行的邮箱采集提示阻塞进程 |

> 部署平台：Streamlit Community Cloud（连接 GitHub，推送 `main` 即自动重新部署）。

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
