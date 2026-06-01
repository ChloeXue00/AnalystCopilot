# AnalystCopilot

> 把一摞研报变成可问答的知识库——几秒钟定位答案，让案头研究从「读一整天」变成「问一句话」。

一个面向行业研究 / 投研场景的 RAG 问答助手：上传 PDF 研报，用自然语言提问，由 Claude 基于报告内容回答并**标注来源**，拒绝凭空编造。

---

## 它解决什么问题

分析师做案头研究（desk research）时，最耗时的不是思考，而是**在几十份 PDF 里翻找一个数字、一句结论**。AnalystCopilot 把这件事自动化：

- **找得快**：语义检索，问"芯片产业前景"也能命中写着"半导体行业"的段落，不必猜关键词
- **答得准**：回答严格基于上传的资料，并标注「来源：《某报告》第 X 段」
- **不瞎编**：资料里没有的，直接回答"根据现有资料无法回答"，而不是用通用知识糊弄

---

## 这个项目里做了哪些「产品决策」

> 这一节是这个仓库的重点——RAG 搭起来不难，难的是发现并解决真实场景里的缺陷。以下每一条都对应一个被实测验证过的问题。

| 发现的问题 | 根因 | 采取的方案 | 效果 |
|-----------|------|-----------|------|
| 研报里的**财务表格丢失** | `extract_text` 把表格压平成一团数字，行列对应关系全断 | 表格单独抽取 → 转 Markdown 整块保留，不参与字符切割 | 表格变成可检索的结构化单元 |
| 表格里混入大量**图表噪声** | pdfplumber 把柱状图/折线图误判成表格，抽出一堆空格和坐标碎片 | 加表格**质量过滤**（非空率、有效字符数启发式） | 真实报告中过滤掉 **75%** 的垃圾表格块（44→11） |
| 多轮对话**指代检索失效** | 追问"那它的毛利率呢"，检索系统不知道"它"指谁 | 检索前用 LLM 做 **query rewriting**，补全指代 | "它"→具体主体，检索命中率提升 |
| 库里没相关内容时**硬塞垃圾→幻觉** | 无脑返回 top-k，哪怕全不相关 | 加**相关度阈值**，距离过大的块直接丢弃 | 无关问题触发"无法回答"而非编造 |
| 单阶段向量检索**精度有限** | 向量检索快但粗 | **两阶段检索**：向量粗筛 top-20 → cross-encoder 精排 top-5 | 召回与精度兼得（可选开关） |
| 改了参数**不知道是否变好** | 全凭感觉调 chunk_size / top_k | 写了 **recall@k / precision@k 评估脚本** | 任何优化都能用数字验证 |

> 工程上还踩过一个真实的坑：`chromadb` 在本机原生模块崩溃（segfault），最终**用纯 numpy 自实现了一个轻量向量库**替代——零原生依赖、永不崩溃，且检索逻辑完全透明可读（见 `utils/embedder.py`）。

---

## 技术架构

```
PDF 上传
   ↓
pdfplumber 解析：正文 / 表格分离，表格转 Markdown          (utils/pdf_parser.py)
   ↓
按段落切分（~400 字，50 字重叠）；表格整块保留
   ↓
sentence-transformers 向量化（多语言 MiniLM，384 维）       (utils/embedder.py)
   ↓
纯 numpy 本地向量库（pickle 持久化，替代 chromadb）
   ↓
用户提问 → [可选] LLM 改写 → 向量检索粗筛 + 距离阈值过滤
   ↓
[可选] cross-encoder 重排，精选 top-k                        (rerank)
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
│   ├── embedder.py         # 向量化 + 纯 numpy 向量库 + cross-encoder 重排
│   └── chat.py             # Claude API 调用、严格提示词、query rewriting
├── eval_retrieval.py       # 检索质量评估（recall@k / precision@k）
├── requirements.txt
├── start_app.bat           # Windows 一键启动
└── README.md
```

---

## 快速开始

```bash
# 1. 安装依赖（torch 较大，可用国内镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置 API Key
#   Windows PowerShell:
$env:ANTHROPIC_API_KEY="sk-ant-xxxxxx"
#   macOS / Linux:
export ANTHROPIC_API_KEY="sk-ant-xxxxxx"

# 3. 启动（也可在 Windows 直接双击 start_app.bat）
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。也可在页面左侧直接填入 API Key。

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

Streamlit · sentence-transformers (MiniLM) · Claude API · pdfplumber · numpy

## 已知限制

- 复杂多层表头 / 无边框表格的解析仍不完美（pdfplumber 局限）
- 向量库存于本地文件，未做多用户隔离
- "全文共有几个表格"这类需纵览全文的问题，RAG 天然不擅长（适合"关于 X 的内容是什么"）
