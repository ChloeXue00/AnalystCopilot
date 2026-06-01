# 行研知识库助手

基于 RAG（检索增强生成）架构的行业研究报告问答系统。上传 PDF 研究报告，用自然语言提问，Claude 基于报告内容精准回答并标注来源。

## 技术架构

```
PDF 上传
   ↓
pdfplumber 解析文本
   ↓
按段落切分（300-500字，50字重叠）
   ↓
sentence-transformers 向量化
   ↓
ChromaDB 本地持久化存储
   ↓
用户提问 → 余弦相似度检索 Top-5 段落
   ↓
Claude API（claude-sonnet-4-20250514）生成回答
   ↓
返回答案 + 来源标注
```

## 文件结构

```
rag_assistant/
├── app.py                  # Streamlit 主程序
├── utils/
│   ├── __init__.py
│   ├── pdf_parser.py       # PDF 解析与文本切分
│   ├── embedder.py         # 向量化 & ChromaDB 操作
│   └── chat.py             # Claude API 调用
├── requirements.txt
├── README.md
└── chroma_db/              # 运行后自动创建，本地向量库
```

## 安装步骤

### 1. 克隆项目并进入目录

```bash
cd rag_assistant
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n rag python=3.10 -y
conda activate rag

# 或使用 venv
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：`torch` 较大（约 2GB），建议使用国内镜像加速：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 4. 配置 API Key

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-xxxxxx"

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-xxxxxx"

# Windows CMD
set ANTHROPIC_API_KEY=sk-ant-xxxxxx
```

或者创建 `.env` 文件（需额外安装 `python-dotenv`）：
```
ANTHROPIC_API_KEY=sk-ant-xxxxxx
```

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`。

## 使用说明

1. **上传报告**：点击左侧「上传研究报告」区域，选择一个或多个 PDF 文件
2. **等待处理**：系统自动解析文本、切块、向量化（首次加载嵌入模型约 1-2 分钟）
3. **开始提问**：在底部输入框输入问题，如：
   - "这份报告的核心结论是什么？"
   - "行业市场规模有多大？增速如何？"
   - "主要竞争对手有哪些？各自优势是什么？"
4. **查看来源**：点击「查看参考来源」展开，查看检索到的原文段落
5. **删除文件**：点击文件列表右侧的 🗑️ 按钮删除单个文件的向量数据
6. **清空对话**：点击「清空对话历史」开始新一轮对话

## 常见问题

**Q: 首次启动很慢？**
A: 需要下载 `paraphrase-multilingual-MiniLM-L12-v2` 模型（约 400MB），之后从缓存加载（几秒内）。

**Q: 如何修改块大小等参数？**
A: 编辑 `app.py` 顶部的常量配置区域：
```python
CHUNK_SIZE = 400   # 每块字符数
OVERLAP = 50       # 重叠字符数
TOP_K = 5          # 检索段落数
```

**Q: 向量数据存在哪里？**
A: 存储在项目根目录的 `./chroma_db/` 文件夹，重启应用后数据不丢失。

**Q: 支持哪些语言的 PDF？**
A: 嵌入模型 `paraphrase-multilingual-MiniLM-L12-v2` 支持 50+ 种语言，中英文均可。

**Q: CPU 能运行吗？**
A: 可以。sentence-transformers 支持纯 CPU 推理，速度稍慢但完全可用。
