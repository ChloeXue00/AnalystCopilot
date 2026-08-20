"""轻量证据图：把实体共现关系与原始 PDF 片段绑定，用于两跳检索扩展。

这不是一个通用知识图谱平台。它只服务于当前会话上传的研报：
- 节点是从文本中可解释地抽取出的公司、行业、指标和英文术语；
- 边表示两个实体在同一原文片段中共同出现；
- 每条边都保留 source/chunk_id/text，回答仍可回到原文引用。
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict, deque
from itertools import combinations
from typing import Dict, Iterable, List, Set, Tuple


GRAPH_DIR = "./graph_store"
MAX_ENTITIES_PER_CHUNK = 12

FINANCE_TERMS = {
    "收入", "营收", "净利润", "毛利率", "净利率", "现金流", "市占率", "市场份额",
    "研发费用", "资本开支", "客户", "供应商", "竞争对手", "风险", "增长率", "同比",
    "环比", "估值", "成本", "价格", "销量", "产能", "需求", "供给", "政策",
}
EN_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "were", "was", "are",
    "report", "page", "table", "figure", "source", "data", "year", "million", "billion",
}


def _graph_path(collection_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
    return os.path.join(GRAPH_DIR, f"{safe}.json")


def extract_entities(text: str) -> List[str]:
    """用透明的规则抽取适合研报检索的实体，避免额外模型调用与上传成本。"""
    found: List[str] = []
    found.extend(re.findall(r"《([^》]{2,30})》", text))
    found.extend(re.findall(
        r"([\u4e00-\u9fffA-Za-z0-9·]{1,20}?(?:公司|集团|股份|银行|科技|平台|行业|市场|业务|产品|模型))",
        text,
    ))
    found.extend(term for term in FINANCE_TERMS if term in text)
    found.extend(
        token for token in re.findall(r"\b[A-Za-z][A-Za-z0-9&.-]{1,24}\b", text)
        if token.lower() not in EN_STOPWORDS and (token.isupper() or any(c.isdigit() for c in token))
    )

    unique: List[str] = []
    seen: Set[str] = set()
    for entity in found:
        normalized = entity.strip(" ·,，。:：;；()（）[]【】")
        key = normalized.casefold()
        if len(normalized) >= 2 and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique[:MAX_ENTITIES_PER_CHUNK]


def build_or_update_graph(chunks: List[Dict], collection_name: str) -> Dict:
    """将新片段写入会话级证据图；同一 source/chunk_id 会覆盖旧记录。"""
    graph = load_graph(collection_name)
    records = {
        f"{r['source']}::{r['chunk_id']}": r
        for r in graph.get("records", [])
    }
    for chunk in chunks:
        entities = extract_entities(chunk["text"])
        key = f"{chunk['source']}::{chunk['chunk_id']}"
        records[key] = {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "entities": entities,
        }
    graph = {"version": 1, "records": list(records.values())}
    os.makedirs(GRAPH_DIR, exist_ok=True)
    with open(_graph_path(collection_name), "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
    return graph


def load_graph(collection_name: str) -> Dict:
    path = _graph_path(collection_name)
    if not os.path.exists(path):
        return {"version": 1, "records": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return {"version": 1, "records": []}


def delete_source_from_graph(source_name: str, collection_name: str) -> int:
    graph = load_graph(collection_name)
    graph["records"] = [r for r in graph.get("records", []) if r.get("source") != source_name]
    os.makedirs(GRAPH_DIR, exist_ok=True)
    with open(_graph_path(collection_name), "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
    return len(graph["records"])


def should_use_graph(query: str) -> bool:
    """关系、因果、传导和跨主体比较问题才触发图扩展。"""
    signals = (
        "关系", "关联", "影响", "导致", "原因", "为什么", "如何传导", "通过什么",
        "上下游", "供应链", "路径", "链条", "间接", "共同", "对比", "比较",
        "relationship", "impact", "cause", "why", "upstream", "downstream", "compare",
    )
    lowered = query.casefold()
    return any(signal in lowered for signal in signals)


def _indexes(records: Iterable[Dict]) -> Tuple[Dict[str, Set[str]], Dict[str, Dict], Dict[str, Set[str]]]:
    entity_records: Dict[str, Set[str]] = defaultdict(set)
    record_map: Dict[str, Dict] = {}
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for record in records:
        rid = f"{record['source']}::{record['chunk_id']}"
        record_map[rid] = record
        entities = record.get("entities", [])
        for entity in entities:
            entity_records[entity.casefold()].add(rid)
        for left, right in combinations(entities, 2):
            adjacency[left.casefold()].add(right.casefold())
            adjacency[right.casefold()].add(left.casefold())
    return entity_records, record_map, adjacency


def query_graph(
    query: str,
    collection_name: str,
    vector_candidates: List[Dict],
    max_hops: int = 2,
    top_k: int = 5,
) -> Tuple[List[Dict], List[str]]:
    """从查询实体/向量候选实体出发做最多两跳 BFS，返回证据片段与可展示路径。"""
    graph = load_graph(collection_name)
    entity_records, record_map, adjacency = _indexes(graph.get("records", []))
    if not record_map:
        return [], []

    query_entities = [e.casefold() for e in extract_entities(query) if e.casefold() in entity_records]
    if not query_entities:
        candidate_keys = {
            f"{c.get('source')}::{c.get('chunk_id')}" for c in vector_candidates[:3]
        }
        query_entities = [
            e.casefold() for rid in candidate_keys for e in record_map.get(rid, {}).get("entities", [])
        ][:4]
    if not query_entities:
        return [], []

    queue = deque((entity, 0, [entity]) for entity in query_entities)
    best_depth: Dict[str, int] = {entity: 0 for entity in query_entities}
    evidence: Dict[str, Tuple[int, str]] = {}
    paths: List[str] = []

    while queue:
        entity, depth, path = queue.popleft()
        for rid in entity_records.get(entity, set()):
            previous = evidence.get(rid)
            if previous is None or depth < previous[0]:
                evidence[rid] = (depth, " → ".join(path))
        if depth >= max_hops:
            continue
        for neighbor in adjacency.get(entity, set()):
            next_depth = depth + 1
            if next_depth < best_depth.get(neighbor, max_hops + 1):
                best_depth[neighbor] = next_depth
                next_path = path + [neighbor]
                queue.append((neighbor, next_depth, next_path))
                if len(next_path) >= 2:
                    paths.append(" → ".join(next_path))

    ranked = sorted(evidence.items(), key=lambda item: (item[1][0], item[0]))[:top_k]
    chunks: List[Dict] = []
    for rid, (depth, path_text) in ranked:
        record = record_map[rid]
        chunks.append({
            "text": record["text"],
            "source": record["source"],
            "chunk_id": record["chunk_id"],
            "distance": round(0.25 + depth * 0.12, 4),
            "retrieval_mode": "graph",
            "graph_path": path_text,
        })
    return chunks, list(dict.fromkeys(paths))[:5]


def merge_evidence(vector_chunks: List[Dict], graph_chunks: List[Dict], top_k: int) -> List[Dict]:
    """保留向量排序，同时加入未重复的图证据，且总上下文受 top_k 限制。"""
    merged: List[Dict] = []
    seen: Set[Tuple[str, int]] = set()
    for chunk in graph_chunks + vector_chunks:
        key = (chunk.get("source", ""), int(chunk.get("chunk_id", -1)))
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= top_k:
            break
    return merged
