"""
pdf_parser.py
负责 PDF 文件的读取、文本提取和段落切分。
使用 pdfplumber 解析，支持中英文混合内容。
"""

import pdfplumber
import re
from pathlib import Path
from typing import List, Dict


def extract_text_from_pdf(file_path: str) -> str:
    """
    从 PDF 文件中提取全部文本。

    Args:
        file_path: PDF 文件的本地路径

    Returns:
        提取到的纯文本字符串（保留换行信息用于后续清洗）
    """
    full_text = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                # 在每页文本前加页码标记，方便后续溯源
                full_text.append(f"[第{page_num}页]\n{text}")
    return "\n".join(full_text)


def clean_text(text: str) -> str:
    """
    清洗提取到的原始文本：
    - 去掉多余空白行
    - 合并因 PDF 排版断开的同一段落（行末无标点的短行拼接到下一行）
    - 保留段落分隔符

    Args:
        text: 原始提取文本

    Returns:
        清洗后的文本
    """
    # 把 Windows 换行统一成 Unix 换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 超过两个连续空行压缩成两个
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去掉每行首尾多余空格
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    source_name: str = "unknown",
) -> List[Dict]:
    """
    将长文本按字符数切分成带重叠的文本块（chunks）。

    策略：
    1. 优先按双换行（自然段落边界）切割
    2. 若某段落超过 chunk_size，再按字符硬切
    3. 相邻块之间保留 overlap 个字符的重叠，保证上下文连贯

    Args:
        text:        清洗后的完整文本
        chunk_size:  每块目标字符数（300-500 之间）
        overlap:     相邻块重叠字符数
        source_name: 来源文件名，写入 metadata 用于溯源

    Returns:
        List[Dict]，每个元素包含:
            - text:    该块文本内容
            - source:  来源文件名
            - chunk_id: 块的顺序编号（从 0 开始）
    """
    # 先按自然段落（双换行）分割
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 把段落逐步拼成满足 chunk_size 的块
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 若单段落就已经超过 chunk_size，先将当前 chunk 保存，再切割该段落
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # 硬切超长段落
            sub_chunks = _hard_split(para, chunk_size, overlap)
            chunks.extend(sub_chunks)
            continue

        # 判断加入当前段落后是否超出 chunk_size
        tentative = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        if len(tentative) <= chunk_size:
            current_chunk = tentative
        else:
            # 保存当前 chunk，开始新 chunk（带 overlap）
            if current_chunk:
                chunks.append(current_chunk.strip())
            # 新 chunk 从上一个 chunk 的结尾 overlap 个字符开始
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = (overlap_text + "\n\n" + para).strip()

    # 保存最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())

    # 组装成带 metadata 的字典列表
    result = []
    for idx, chunk_text in enumerate(chunks):
        if chunk_text:  # 跳过空块
            result.append({
                "text": chunk_text,
                "source": source_name,
                "chunk_id": idx,
            })

    return result


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    对超长文本进行强制按字符数切分（内部辅助函数）。

    Args:
        text:       待切分的长文本
        chunk_size: 每块最大字符数
        overlap:    相邻块重叠字符数

    Returns:
        切分后的文本列表
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 下一块从 (end - overlap) 开始
        start = max(start + 1, end - overlap)
    return chunks


def _is_quality_table(rows: List[List]) -> bool:
    """
    判断 pdfplumber 抽出的"表格"是否是【真表格】，过滤掉垃圾。

    背景：pdfplumber 常把图表(柱状图/折线图)、装饰性框线误判成表格，
    抽出来的是大量空单元格或坐标轴上的散落单字符（如 "M e X 0 E 0"）。
    这些垃圾块入库会污染检索。这里用几条简单启发式规则筛掉它们。

    判定为【真表格】需同时满足：
      1. 至少 2 行 2 列（否则不构成表格）
      2. 非空单元格占比 ≥ 30%（太多空格 → 多半是图表框线误判）
      3. 有足够的"有意义文本"：去掉孤立单字符后，总有效字符数 ≥ 10
         （图表坐标轴抽出来全是 "1.2 0.8 M e X" 这种碎片）

    Args:
        rows: [[cell, ...], ...]
    Returns:
        True=真表格保留，False=垃圾丢弃
    """
    # 清洗成纯字符串二维表
    grid = [[(c or "").strip() for c in row] for row in rows if row]
    if len(grid) < 2:
        return False
    ncol = max((len(r) for r in grid), default=0)
    if ncol < 2:
        return False

    cells = [c for row in grid for c in row]
    total = len(cells)
    nonempty = [c for c in cells if c]

    # 规则2：非空率
    if total == 0 or len(nonempty) / total < 0.30:
        return False

    # 规则3：有意义文本量（长度≥2的单元格才算"有内容"，过滤孤立单字符碎片）
    meaningful_chars = sum(len(c) for c in nonempty if len(c) >= 2)
    if meaningful_chars < 10:
        return False

    return True


def _table_to_markdown(rows: List[List]) -> str:
    """
    把 pdfplumber 抽出的表格（行列二维列表）转成 Markdown 表格字符串。

    为什么转 Markdown：
    - 表头和数据按列对齐，LLM 能正确读懂「某行某列 = 某值」
    - embed 时整行上下文（表头+数值）保留在一起，语义不再糊成一团

    Args:
        rows: [[cell, cell, ...], ...]，单元格可能是 None
    Returns:
        Markdown 表格字符串；空表返回 ""
    """
    # 清洗每个单元格：None→""，去掉换行和首尾空格
    clean = [
        [(cell or "").replace("\n", " ").strip() for cell in row]
        for row in rows if row
    ]
    if not clean:
        return ""

    # 补齐每行列数（有些行单元格缺失）
    ncol = max(len(r) for r in clean)
    clean = [r + [""] * (ncol - len(r)) for r in clean]

    # 第一行当表头，第二行是 Markdown 的分隔线
    lines = ["| " + " | ".join(clean[0]) + " |",
             "| " + " | ".join(["---"] * ncol) + " |"]
    for r in clean[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _extract_page(page) -> tuple:
    """
    从单页中分离「正文」和「表格」。

    Args:
        page: pdfplumber 的 Page 对象
    Returns:
        (prose_text, [markdown_table, ...])
        prose_text: 抠掉表格区域后的正文
        列表元素: 每个表格转好的 Markdown
    """
    tables = page.find_tables()

    # 没有表格：正文就是整页文本
    if not tables:
        return (page.extract_text() or ""), []

    # 有表格：把落在表格框内的字符过滤掉，剩下的才是纯正文
    bboxes = [t.bbox for t in tables]  # (x0, top, x1, bottom)

    def outside_tables(obj) -> bool:
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        for (x0, top, x1, bottom) in bboxes:
            if x0 <= cx <= x1 and top <= cy <= bottom:
                return False  # 在某个表格框内 → 不算正文
        return True

    try:
        prose = page.filter(outside_tables).extract_text() or ""
    except Exception:
        # 过滤偶尔会出问题，兜底用整页文本
        prose = page.extract_text() or ""

    # 只保留通过质量筛选的真表格，丢掉图表/框线误判出的垃圾块
    table_mds = []
    for t in tables:
        rows = t.extract()
        if _is_quality_table(rows):
            table_mds.append(_table_to_markdown(rows))
    return prose, table_mds


def parse_pdf(file_path: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    PDF 解析主入口（表格感知版）：
    逐页分离正文与表格 → 正文清洗+切块 → 表格各自作为完整块 → 统一编号。

    关键改进：表格不再被压平进正文，而是转成 Markdown、整块保留，
    保住研报里最核心的财务/数据表格。

    Args:
        file_path:  PDF 文件路径
        chunk_size: 正文每块目标字符数
        overlap:    正文重叠字符数

    Returns:
        切分好的文本块列表，每块包含 text / source / chunk_id
    """
    source_name = Path(file_path).name

    prose_parts: List[str] = []
    table_blocks: List[tuple] = []  # [(页码, markdown表格), ...]

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            prose, table_mds = _extract_page(page)
            if prose.strip():
                prose_parts.append(f"[第{page_num}页]\n{prose}")
            for tmd in table_mds:
                if tmd.strip():
                    table_blocks.append((page_num, tmd))

    # ① 正文：清洗后按原逻辑切块
    clean = clean_text("\n".join(prose_parts))
    chunks = split_into_chunks(clean, chunk_size=chunk_size, overlap=overlap, source_name=source_name)

    # ② 表格：每个表作为一个完整块，不参与字符切割（切碎就废了）
    #    加【表格】标记，方便 LLM 识别、也方便检索结果里一眼认出
    for page_num, tmd in table_blocks:
        chunks.append({
            "text": f"【表格】（第{page_num}页）\n{tmd}",
            "source": source_name,
            "chunk_id": 0,  # 占位，下面统一重排
        })

    # ③ 统一重排 chunk_id，保证全局连续唯一
    for idx, chunk in enumerate(chunks):
        chunk["chunk_id"] = idx

    return chunks
