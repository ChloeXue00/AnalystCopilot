"""
eval_retrieval.py —— 评估「检索」这一步的质量：recall@k / precision@k

═══════════════════════════════════════════════════════════════════
为什么需要它？
  你改 chunk_size、换嵌入模型、加 rerank……到底变好还是变坏，
  不能靠"感觉"，要靠数字。这个脚本就是那把「尺子」。

核心定义（k = 检索返回前几个块）：
  recall@k    = 命中的相关块数 / 全库相关块总数   →「该找到的，找回了多少」
  precision@k = 命中的相关块数 / k               →「找回的里面，多少是对的」

怎么判断一个块"相关"？
  理想做法是人工标注（最准）。这里用一个轻量近似：
  给每个测试问题配几个「关键词」，块的文本里包含这些关键词就算相关。
  —— 这是人工判断的"自动代理"，够你练手和对比优化效果；
     等你要发论文级别的严谨结论，再换成人工标注的 gold label。

两种运行模式：
  python eval_retrieval.py demo   # 自带迷你语料，绕开 ChromaDB，任何环境都能跑
  python eval_retrieval.py        # 接你真实的 ChromaDB 知识库
═══════════════════════════════════════════════════════════════════
"""

import sys
from typing import List, Dict, Callable

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台防中文乱码


# ─────────────────────────────────────────────────────────────────
# 1) 相关性判定：块的文本是否包含问题要求的全部关键词
# ─────────────────────────────────────────────────────────────────
def is_relevant(text: str, keywords: List[str]) -> bool:
    """只要 text 同时包含 keywords 里的所有词，就判为"相关"。"""
    return all(kw in text for kw in keywords)


# ─────────────────────────────────────────────────────────────────
# 2) 指标函数（纯计算，不依赖任何检索后端 —— 这是最该看懂的核心）
#    输入 flags：按检索排名排好序的 0/1 列表，1 = 该位置的块是相关的
# ─────────────────────────────────────────────────────────────────
def recall_at_k(flags: List[int], total_relevant: int, k: int) -> float:
    """recall@k = 前 k 个里命中的相关块数 / 全库相关块总数"""
    if total_relevant == 0:
        return 0.0
    hits = sum(flags[:k])
    return hits / total_relevant


def precision_at_k(flags: List[int], k: int) -> float:
    """precision@k = 前 k 个里命中的相关块数 / k"""
    if k == 0:
        return 0.0
    hits = sum(flags[:k])
    return hits / k


# ─────────────────────────────────────────────────────────────────
# 3) 评估主循环（后端无关：检索这一步由外部传入的 retrieve_fn 完成）
#    retrieve_fn(question, n) -> 按相关度排好序的"块文本"列表（长度 n）
#    all_texts -> 全库所有块的文本（用来数"全库相关块总数"）
# ─────────────────────────────────────────────────────────────────
def evaluate(
    eval_set: List[Dict],
    retrieve_fn: Callable[[str, int], List[str]],
    all_texts: List[str],
    k_values: List[int],
) -> None:
    max_k = max(k_values)
    # 累加每个 k 的指标，最后求平均
    sum_recall = {k: 0.0 for k in k_values}
    sum_prec = {k: 0.0 for k in k_values}

    print("=" * 70)
    print("逐题明细")
    print("=" * 70)

    for item in eval_set:
        q = item["question"]
        kws = item["keywords"]

        # 全库里到底有几个块是相关的（recall 的分母）
        total_relevant = sum(is_relevant(t, kws) for t in all_texts)

        # 跑检索，取前 max_k 个，转成 0/1 命中标记
        ranked_texts = retrieve_fn(q, max_k)
        flags = [1 if is_relevant(t, kws) else 0 for t in ranked_texts]

        print(f"\n问题：{q}")
        print(f"  关键词：{kws}  |  全库相关块总数：{total_relevant}")
        print(f"  检索命中标记(前{max_k})：{flags}   (1=相关, 0=不相关)")
        for k in k_values:
            r = recall_at_k(flags, total_relevant, k)
            p = precision_at_k(flags, k)
            sum_recall[k] += r
            sum_prec[k] += p
            print(f"    @{k}:  recall={r:.2f}   precision={p:.2f}")

    # ── 汇总：所有问题求平均 ──
    n = len(eval_set)
    print("\n" + "=" * 70)
    print(f"平均指标（{n} 个问题）")
    print("=" * 70)
    print(f"{'k':>4} | {'recall@k':>10} | {'precision@k':>12}")
    print("-" * 34)
    for k in k_values:
        print(f"{k:>4} | {sum_recall[k] / n:>10.3f} | {sum_prec[k] / n:>12.3f}")
    print("\n看点：k 增大时 recall 通常↑、precision 通常↓ —— 这就是召回/精确的权衡。")
    print("rerank 的意义：让你能用大 k 保 recall，再把对的顶到前面、取小 n 保 precision。")


# ═════════════════════════════════════════════════════════════════
# 模式 A：demo —— 自带迷你语料 + sentence-transformers 直接检索（绕开 ChromaDB）
# ═════════════════════════════════════════════════════════════════
def run_demo() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # 迷你"行研"语料：每条相当于一个 chunk
    corpus = [
        "半导体行业2024年保持高速增长，全球销售额同比上升超过20%。",   # 0
        "人工智能芯片需求旺盛，是带动半导体增长的核心动力。",           # 1
        "新能源汽车销量持续攀升，动力电池产能紧张。",                   # 2
        "光伏组件价格持续下行，行业进入整合出清阶段。",                 # 3
        "存储芯片价格触底反弹，预计明年行业景气度回升。",               # 4
        "公司治理结构完善，董事会下设审计、薪酬等四个专门委员会。",     # 5
    ]
    eval_set = [
        {"question": "半导体行业今年增长情况如何？", "keywords": ["半导体"]},
        {"question": "存储芯片价格走势怎么样？",       "keywords": ["存储", "芯片"]},
        {"question": "新能源车的电池供应情况？",       "keywords": ["电池"]},
        {"question": "人工智能对芯片的拉动？",         "keywords": ["人工智能"]},
    ]

    print("加载模型（与 app.py 同款 MiniLM，本地缓存）...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    corpus_vecs = model.encode(corpus)
    

    def retrieve_fn(question: str, n: int) -> List[str]:
        qv = model.encode([question])[0]
        # 余弦相似度 = 归一化后的点积
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        scores = [cos(qv, cv) for cv in corpus_vecs]
        order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
        return [corpus[i] for i in order[:n]]

    evaluate(eval_set, retrieve_fn, all_texts=corpus, k_values=[1, 3, 5])


# ═════════════════════════════════════════════════════════════════
# 模式 B：real —— 接你真实的本地向量库（utils/embedder.py 的 numpy 库）
# ═════════════════════════════════════════════════════════════════
def run_real() -> None:
    from utils.embedder import load_embedding_model, query_similar_chunks, get_all_documents

    COLLECTION = "rag_docs"
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    # ⚠️ 改成你知识库里真实文档相关的问题和关键词！下面是占位示例。
    eval_set = [
        {"question": "这份报告的核心结论是什么？", "keywords": ["结论"]},
        {"question": "市场规模有多大？",           "keywords": ["规模"]},
    ]

    # 取全库所有块文本，用于数"全库相关块总数"
    all_texts = get_all_documents(COLLECTION)
    if not all_texts:
        print("知识库为空，请先在 app 里上传 PDF 再来评估。")
        return
    print(f"知识库共 {len(all_texts)} 个块。")

    model = load_embedding_model(EMBEDDING_MODEL)

    # ─── 一个开关，跑两次对比 rerank 效果 ───────────────────────
    USE_RERANK = False                       # 改 True 启用精排（需先下载好 reranker）
    RERANK_MODEL = "BAAI/bge-reranker-base"  # 若用 ModelScope 下到本地，改成本地路径

    if USE_RERANK:
        # 带 rerank：向量粗筛 top-20 → cross-encoder 精排取 top-n
        from utils.embedder import load_reranker, rerank_chunks
        reranker = load_reranker(RERANK_MODEL)

        def retrieve_fn(question: str, n: int) -> List[str]:
            cands = query_similar_chunks(question, model, top_k=20, collection_name=COLLECTION)
            return [c["text"] for c in rerank_chunks(question, cands, reranker, top_n=n)]
    else:
        # 基线：纯向量检索 top-n
        def retrieve_fn(question: str, n: int) -> List[str]:
            chunks = query_similar_chunks(question, model, top_k=n, collection_name=COLLECTION)
            return [c["text"] for c in chunks]

    print(f"\n>>> 当前模式：{'带 rerank 精排' if USE_RERANK else '基线（纯向量）'}\n")
    evaluate(eval_set, retrieve_fn, all_texts=all_texts, k_values=[1, 3, 5, 10])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    if mode == "demo":
        run_demo()
    else:
        run_real()
