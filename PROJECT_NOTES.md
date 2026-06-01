# 行研知识库助手 · 项目复盘 & 面试速查

> 一句话定位：基于 **RAG** 的行业研究报告问答系统。上传 PDF 研报，自然语言提问，
> Claude 基于报告**原文**作答并**标注来源**。解决投研"找信息慢 / 多文档对比难 / 大模型幻觉无法溯源"三大痛点。

---

## 0. 一分钟讲清楚（电梯陈述）

> "我做了一个面向行研/投研的 RAG 问答助手。痛点是研究员每天读大量 PDF 研报，
> 关键词搜不到同义词、跨报告对比要手动切、直接问大模型又会幻觉且无法溯源。
> 我用 **pdfplumber 解析 + 语义切块 + sentence-transformers 向量化 + ChromaDB 检索 + Claude 生成**，
> 实现了**语义检索 + 强制溯源**。整个过程先做 MVP 跑通，再在自测中修了三个工程坑，
> 最后针对专业场景补了溯源和知识库维护，并搭了离线评测脚本用 Recall@5 / Faithfulness 量化迭代。"

---

## 1. 技术链路（5 个环节）

| 环节 | 技术 | 为什么是它 |
|---|---|---|
| PDF 解析 | `pdfplumber` | 比 PyPDF2 更好保留版式/表格，中英文混排稳 |
| 文本切割 | 段落优先 + 字符兜底 + 重叠 | 语义完整 + 长度可控（见 §4） |
| 向量化 | `paraphrase-multilingual-MiniLM-L12-v2`（MiniLM，384维，多语言，本地免费） | 中英文都行，CPU 可跑 |
| 存储/检索 | `ChromaDB` 本地持久化 + HNSW + 余弦相似度 | 零运维、落盘、自带近似最近邻 |
| 生成 | Claude `claude-sonnet-4-20250514` | 长上下文、中文强、指令遵循好 |

**两个底座要分清**：检索底座 = MiniLM；生成底座 = Claude Sonnet 4。

**RAG 本质**：把"让模型记住一切"转成"**先检索后生成**"——
离线：切块→向量→入库；在线：问题向量化→余弦相似度取 Top-K→塞进 prompt→Claude 仅据此作答。
余弦相似度比的是**语义方向**，所以"营收"能匹配"销售收入"，解决关键词搜不到同义词的痛点。

---

## 2. 关键参数（怎么定的）

集中在 [app.py:31-37](app.py#L31-L37)

| 参数 | 值 | 依据 |
|---|---|---|
| `CHUNK_SIZE` | 400 字符 | 中文 ≈ 一个完整观点段；太小语义不全，太大检索精度降 |
| `OVERLAP` | 50 字符 | ≈12.5%（经验 10-20%），接住跨边界句子 |
| `TOP_K` | 5 | 5×400≈2000字上下文，覆盖够又不淹没；太大引噪声 |
| `MAX_TOKENS` | 2048 | 回答上限，研报问答几百字够用 |
| `MAX_HISTORY_TURNS` | 6 | 滑动窗口，防多轮 token 无限膨胀 |
| 距离度量 | cosine | 语义检索标准，对向量长度不敏感 |

**上下文窗口**：实际喂给 Claude ≈ system(~150字) + Top5×400字(~2000) + 历史 + 问题，
通常几千 token，**远低于 Claude 200K 上限**——RAG 价值正在此，不必塞整本 PDF。

---

## 3. 数据清洗（代码做的，不是模型做的）→ [pdf_parser.py](utils/pdf_parser.py)

1. **提取** `extract_text_from_pdf`：逐页抽字，每页打 `[第X页]` 标记便于溯源
2. **清洗** `clean_text`：统一换行、压缩空行、去行首尾空格 → 去脏空白避免污染向量
3. **切块** `split_into_chunks`：三层递进（见 §4）

**局限**：纯文字清洗，**丢表格结构**（研报数字常藏表格里）→ 改进方向 `extract_tables()` 结构化。

---

## 4. 文本切割（三层递进）→ [pdf_parser.py:59](utils/pdf_parser.py#L59)

1. **段落优先**：按 `\n\n+` 切自然段，尊重语义边界
2. **贪心累积**：小段拼到接近 400 字封块
3. **超长硬切**：单段超 400 用 `_hard_split` 按字符切
4. **重叠 50 字**：新块带上一块尾部 50 字，防答案跨边界被切散

**为什么**：纯固定长度会劈断句子；纯按段落则长度不可控。混合策略兼顾**语义完整 + 长度可控**。

---

## 5. 工程改进（按开发阶段讲，重点！）

```
阶段1 MVP            阶段2 自测踩坑              阶段3 打磨
 └ 批量 encode        └ cache_resource 缓存模型   └ 溯源 UI + 相关度
                     └ 持久化 PersistentClient   └ 按文件删除
                     └ sha256 去重
```

### 阶段1 MVP
- **批量 encode** [embedder.py:111](utils/embedder.py#L111)
  - why：逐条 `encode` 调度开销大、利用率低
  - how：收集成 list 一次 `encode(texts)`，内部自动分 batch
  - when：第一版就该这么写（基础素养）

### 阶段2 自测踩坑（"跑起来才暴露"的连环坑）
- **@st.cache_resource 缓存模型** [app.py:55](app.py#L55)
  - why：Streamlit 每次交互 rerun → 第一版每次重载 400MB 模型等 1-2 分钟
  - how：装饰器保证整 session 只加载一次 → 秒级
  - when：第一次点界面就崩溃式慢
- **持久化 PersistentClient** [embedder.py:19](utils/embedder.py#L19)
  - why：默认内存模式，重启丢库要重新 embed
  - how：`PersistentClient(path="./chroma_db")` 落盘
  - when：关掉重开发现数据没了
- **sha256 确定性 doc_id + upsert 去重** [embedder.py:63](utils/embedder.py#L63)
  - why：持久化后重复上传产生重复向量，挤占 Top-5
  - how：`sha256(文件名)[:8]+"_chunk_"+序号` 同块永远同 ID，配 upsert
  - when：持久化做完测重复上传时发现（与上条连环）

### 阶段3 打磨（针对"专业用户"）
- **溯源 UI + 相关度百分比** [chat.py:147](utils/chat.py#L147)
  - why：投研最忌无法验证的答案，必须核对来源
  - how：system prompt 强制标来源 + 展开面板列原文预览 + `(1-distance)×100%` 相关度
  - when：功能可用后做可信度增强（体现懂业务）
- **按文件删除** [embedder.py:198](utils/embedder.py#L198)
  - why：过期/传错报告会污染检索
  - how：按 `source` metadata `collection.delete(where=...)` 精准删
  - when：长期使用的可维护性需求

**面试叙事**：先 MVP 跑通 → 自测踩三个坑逐个定位修复 → 针对行研场景补溯源和维护
→ 从"能用"到"敢用、好用"。

---

## 6. 评测体系（怎么证明它好）→ [evaluate.py](evaluate.py)

**检索层（免费本地）**
- **Recall@K**（最关键）：正确来源段被召回的比例。
  例：答案分布在 5 个正确段，Top-5 命中 3 个 → Recall@5 = 3/5 = **0.6**（漏了 40%，偏低，目标 >0.8）
- Hit Rate：至少命中一段的问题比例
- MRR：首个命中排名倒数均值（越靠前越高）

**生成层（调 Claude，RAGAS 思路）**
- **Faithfulness 忠实度**（投研最看重）：答案是否只依据检索资料（防幻觉），LLM-as-judge 打分
- Answer Relevancy / Context Precision / Recall

**系统层**：端到端延迟、单次成本（token×单价，单次通常 <1 美分）、来源点击率/追问率

**用法**：标 20-30 条黄金问答 → 跑基线 → 改参数（top_k / chunk_size）再跑 → 对比数字。
`top_k` 是设定的检索数（输入）；`Recall@k` 是衡量漏没漏（输出）。调大 top_k 通常提 Recall，
但 precision 降、成本升 → 用 evaluate.py 找平衡点。

---

## 7. 需求优先级（RICE = Reach×Impact×Confidence÷Effort）

| 需求 | RICE | 结论 |
|---|---|---|
| 修 API Key 崩溃 bug | 极高 | 立刻（已修） |
| 加 reranker 提准 | 高 | 优先 |
| 多轮历史压缩 | 中高 | 已做 |
| 表格结构化提取 | 中 | 排期 |
| 流式输出 | 中 | 可做 |

口诀：**先修崩溃 → 再做高 RICE 的检索质量 → 体验类穿插**，每次投入用 §6 指标验证。

---

## 8. 常见问题 & Fix

**代码已修**
- API Key 输入框 `NameError`（用错变量名）→ 改 `api_key_input` ✅ [app.py:113](app.py#L113)
- 多轮历史无限增长 → 滑动窗口 `MAX_HISTORY_TURNS=6` ✅ [chat.py:9](utils/chat.py#L9)

**RAG 通病**
| 问题 | Fix |
|---|---|
| 检索召回低 | rerank、混合检索(向量+BM25)、query 改写 |
| 表格/数字丢失 | `extract_tables()` 结构化 |
| 跨块切散 | 调大 overlap、父子块检索 |
| 幻觉残留 | 强 prompt + 答案与检索块一致性校验 |
| 首次加载慢 | cache_resource（已做）、预下载 |

---

## 9. 演进路线（被问"还能怎么做"）

```
现在:  问 → [检索1次] → 答                         一问一答 RAG
 ↓ 加规划/反思
第一档: 问 → [拆解→多次检索→反思→综合]             Agentic RAG
 ↓ 加工具
第二档: 问 → Agent ⇄ {检索/计算/联网/表格对比}      Tool Use / Function Calling
 ↓ 加分工+记忆
第三档: 问 → [Planner→Retriever→Analyst→Writer]+Memory   Multi-Agent
```

**关键认知**：LLM 不擅长的（精确计算、实时数据、确定性操作）都做成工具交出去。
**成熟表态**：Agent 更强但更慢/贵/难调试；简单问答用单次 RAG 反而更好；
每加一档都要用 Eval 验证真的变好，避免过度工程。

---

## 10. LLM 关键名词地图（"X 对应解决什么"）

- **知道得更多/更新/不瞎编**：RAG(溯源,不改权重) / 微调 SFT(改权重学领域) / Embedding / Reranker
- **答得合心意（对齐）**：RLHF / DPO(简化版) / LoRA·QLoRA·PEFT(低成本微调) / 蒸馏(降本)
- **会动手（Agent）**：Agent / Function Calling / ReAct / MCP / Multi-agent
- **想得清楚**：Prompt Engineering / Few-shot / CoT / System Prompt
- **跑得快省**：量化 / KV Cache / MoE / Context Window / Streaming
- **安全评测**：Hallucination(问题) / Eval·RAGAS / Guardrails / Temperature·Top-p

**主线**：让 LLM 落地 = 知道得对(RAG/微调) → 答得合心意(RLHF) → 会干活(Agent) → 跑得起(量化) → Eval 证明变好。
本项目踩在第一格(RAG 溯源)。

---

## 计费备注（别混）
- **VSCode 里的 Claude Code** = 走 **Claude Max 订阅额度**（Auth: Claude AI），不花 API 钱
- **本项目里的 Claude** = 走 **API 按量付费**（需 `sk-ant-` key），与订阅是两本账
- 嵌入用本地 MiniLM，**免费**；项目唯一花 API 钱的是 Claude 生成那一步（单次 <1 美分）
