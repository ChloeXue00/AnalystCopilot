"""
embedder.py
负责：
1. 用 sentence-transformers 将文本块向量化
2. 一个轻量的【纯 Python / numpy 向量库】（替代 ChromaDB）

为什么不用 ChromaDB？
  本机上 chromadb 的原生(Rust/onnxruntime)模块在做向量运算时会崩溃(segfault)，
  各版本都试过无法解决。RAG 的检索本质就是「算余弦相似度 + 排序」，
  用 numpy 几十行就能实现，零原生依赖、永不崩溃，而且一切透明可见。

存储格式：每个 collection 一个 pickle 文件，放在 ./vector_store/ 下。
  内容是一个 dict：
    ids        : List[str]            每个块的唯一 id
    embeddings : np.ndarray (N, D)    N 个块的向量
    documents  : List[str]           块的文本
    metadatas  : List[dict]          {"source":..., "chunk_id":...}
"""

import os
import pickle
import hashlib
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# 向量库持久化目录
STORE_DIR = "./vector_store"


# ─────────────────────────────────────────────────────────────────
# 模型加载（与原来一致，sentence-transformers 在本机正常工作）
# ─────────────────────────────────────────────────────────────────
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    加载 sentence-transformers 多语言嵌入模型。
    paraphrase-multilingual-MiniLM-L12-v2 支持中文、英文等 50+ 种语言。
    """
    return SentenceTransformer(model_name)


def load_reranker(model_name: str) -> CrossEncoder:
    """
    加载 cross-encoder 重排模型（第二阶段精排用）。
    与嵌入模型(双塔)不同：它把(问题,文档)拼在一起打分，准但慢，只对粗筛候选跑。
    中文场景推荐 "BAAI/bge-reranker-base"。
    """
    return CrossEncoder(model_name)


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    reranker: CrossEncoder,
    top_n: int = 5,
) -> List[Dict]:
    """
    用 cross-encoder 对粗筛候选重排，返回分数最高的 top_n 个（每块加 rerank_score）。
    两阶段检索的第二阶段：向量粗筛(保 recall) → 这里精排(保 precision)。
    """
    if not chunks:
        return []
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n]


# ─────────────────────────────────────────────────────────────────
# 向量库底层：加载 / 保存 / 唯一 id
# ─────────────────────────────────────────────────────────────────
def _store_path(collection_name: str) -> str:
    return os.path.join(STORE_DIR, f"{collection_name}.pkl")


def _empty_store() -> Dict:
    return {"ids": [], "embeddings": None, "documents": [], "metadatas": []}


def _load_store(collection_name: str) -> Dict:
    """从磁盘读出向量库；不存在则返回空结构。"""
    path = _store_path(collection_name)
    if not os.path.exists(path):
        return _empty_store()
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_store(collection_name: str, store: Dict) -> None:
    """把向量库写回磁盘。"""
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(_store_path(collection_name), "wb") as f:
        pickle.dump(store, f)


def _make_doc_id(source: str, chunk_id: int) -> str:
    """为每个块生成唯一 id：sha256(source)[:8] + _chunk_{chunk_id}，避免重复插入。"""
    hash_prefix = hashlib.sha256(source.encode()).hexdigest()[:8]
    return f"{hash_prefix}_chunk_{chunk_id}"


# ─────────────────────────────────────────────────────────────────
# 写入：向量化 + 存库（按 id 去重，相当于 upsert）
# ─────────────────────────────────────────────────────────────────
def add_chunks_to_db(
    chunks: List[Dict],
    model: SentenceTransformer,
    collection_name: str = "rag_docs",
) -> int:
    """
    将文本块向量化并存入本地向量库。

    步骤：
    1. 批量编码所有 chunk 的文本 → 向量
    2. 按 id 去重合并进已有库（同 id 覆盖更新）
    3. 落盘
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    new_ids = [_make_doc_id(c["source"], c["chunk_id"]) for c in chunks]
    new_metas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]
    new_embs = np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float32)

    store = _load_store(collection_name)

    # 建立 id→行号 索引，实现 upsert（已存在则覆盖，否则追加）
    id_to_row = {doc_id: i for i, doc_id in enumerate(store["ids"])}
    emb = store["embeddings"]

    for i, doc_id in enumerate(new_ids):
        if doc_id in id_to_row:
            row = id_to_row[doc_id]
            store["documents"][row] = texts[i]
            store["metadatas"][row] = new_metas[i]
            emb[row] = new_embs[i]
        else:
            store["ids"].append(doc_id)
            store["documents"].append(texts[i])
            store["metadatas"].append(new_metas[i])
            emb = new_embs[i:i + 1] if emb is None else np.vstack([emb, new_embs[i]])
            id_to_row[doc_id] = len(store["ids"]) - 1

    store["embeddings"] = emb
    _save_store(collection_name, store)
    return len(chunks)


# ─────────────────────────────────────────────────────────────────
# 检索：余弦相似度 + 排序 + 阈值过滤
# ─────────────────────────────────────────────────────────────────
def query_similar_chunks(
    query: str,
    model: SentenceTransformer,
    top_k: int = 5,
    collection_name: str = "rag_docs",
    source_filter: List[str] = None,
    max_distance: float = None,
) -> List[Dict]:
    """
    检索与问题最相关的文本块。

    步骤：
    1. 问题向量化
    2. 和库里所有向量算余弦相似度（distance = 1 - 相似度，越小越相关）
    3. 升序排序取 top_k，再按 max_distance 丢掉太不相关的

    Args 同原版；返回每项含 text / source / chunk_id / distance。
    加 max_distance 后返回数可能少于 top_k（甚至为空）。
    """
    store = _load_store(collection_name)
    emb = store["embeddings"]
    if emb is None or len(store["ids"]) == 0:
        return []

    # 可选：按来源文件过滤（只在指定文件里搜）
    idxs = list(range(len(store["ids"])))
    if source_filter:
        sf = set(source_filter)
        idxs = [i for i in idxs if store["metadatas"][i].get("source") in sf]
        if not idxs:
            return []

    sub_emb = emb[idxs]
    qv = np.asarray(model.encode([query], show_progress_bar=False)[0], dtype=np.float32)

    # 余弦相似度 = 点积 / (各自模长)
    sims = sub_emb @ qv / (np.linalg.norm(sub_emb, axis=1) * np.linalg.norm(qv) + 1e-10)
    distances = 1.0 - sims

    # 按距离升序（最相关在前）
    order = np.argsort(distances)

    output = []
    for j in order[:top_k]:
        i = idxs[j]
        output.append({
            "text": store["documents"][i],
            "source": store["metadatas"][i].get("source", "unknown"),
            "chunk_id": store["metadatas"][i].get("chunk_id", -1),
            "distance": round(float(distances[j]), 4),
        })

    if max_distance is not None:
        output = [c for c in output if c["distance"] <= max_distance]
    return output


# ─────────────────────────────────────────────────────────────────
# 管理：删除 / 列出来源 / 统计
# ─────────────────────────────────────────────────────────────────
def delete_source_from_db(source_name: str, collection_name: str = "rag_docs") -> int:
    """删除某个文件的所有块，返回库中剩余块数。"""
    store = _load_store(collection_name)
    if not store["ids"]:
        return 0

    keep = [i for i, m in enumerate(store["metadatas"]) if m.get("source") != source_name]
    store["ids"] = [store["ids"][i] for i in keep]
    store["documents"] = [store["documents"][i] for i in keep]
    store["metadatas"] = [store["metadatas"][i] for i in keep]
    store["embeddings"] = store["embeddings"][keep] if (keep and store["embeddings"] is not None) else None

    _save_store(collection_name, store)
    return len(store["ids"])


def get_all_sources(collection_name: str = "rag_docs") -> List[str]:
    """列出库中所有已索引的文件名（去重排序）。"""
    store = _load_store(collection_name)
    sources = {m.get("source") for m in store["metadatas"] if m.get("source")}
    return sorted(sources)


def get_chunk_count_per_source(collection_name: str = "rag_docs") -> Dict[str, int]:
    """统计每个文件的块数（UI 展示用）。"""
    store = _load_store(collection_name)
    counts: Dict[str, int] = {}
    for m in store["metadatas"]:
        src = m.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def get_all_documents(collection_name: str = "rag_docs") -> List[str]:
    """返回库中所有块的文本（评估脚本算"全库相关块总数"时用）。"""
    return list(_load_store(collection_name)["documents"])
