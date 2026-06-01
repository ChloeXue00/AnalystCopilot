"""
evaluate.py
行研助手的离线评测脚本（mini RAG 评估框架）。

用途：
    每次调整参数（chunk_size / top_k / 换嵌入模型 / 改 prompt）后，
    跑一遍本脚本，用「黄金问答集」量化检索质量和生成忠实度，
    用真实数字判断改动是变好还是变坏，而不是凭感觉。

两层指标：
    1) 检索层（纯本地、零成本、零幻觉风险）：
       - Hit Rate   ：Top-K 里至少命中 1 个正确来源段的问题比例
       - Recall@K   ：正确来源段被检索回来的比例（最关键）
       - MRR        ：第一个命中段的排名倒数均值（越靠前越好）
    2) 生成层（需调用 Claude，按 --judge 开关，有成本）：
       - Faithfulness：答案是否「只依据检索资料」，由 Claude 当裁判打 0~1 分
                       —— 投研场景最看重，直接对应「防幻觉」。

用法：
    # 1. 先用 app.py 上传过 PDF（向量库已有数据），再编辑 golden_set.json
    # 2. 只跑检索指标（免费）：
    python evaluate.py
    # 3. 同时跑忠实度（调用 Claude，需 ANTHROPIC_API_KEY）：
    python evaluate.py --judge

黄金集格式（golden_set.json）：
    [
      {
        "question": "这份报告预测的 2025 年市场规模是多少？",
        "expected_sources": ["行业报告A.pdf"],          # 必填：答案应来自哪些文件
        "expected_chunk_ids": [12, 13],                  # 可选：精确到段号则指标更严
        "reference_answer": "约 1200 亿元"               # 可选：仅供人工对照
      }
    ]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from utils.embedder import load_embedding_model, query_similar_chunks
from utils.chat import get_anthropic_client, build_rag_context

# 与 app.py 保持一致的默认配置
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CHROMA_COLLECTION = "rag_docs"
TOP_K = 5

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def load_golden_set(path: Path) -> List[Dict[str, Any]]:
    """读取黄金问答集；不存在时生成一个模板并退出。"""
    if not path.exists():
        template = [
            {
                "question": "把这里换成你的真实问题，例如：报告预测的市场规模是多少？",
                "expected_sources": ["你的文件名.pdf"],
                "expected_chunk_ids": [],
                "reference_answer": "（可选）标准答案，仅供人工对照",
            }
        ]
        path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"⚠️  未找到 {path.name}，已生成模板。请填入真实问答后重新运行。")
        sys.exit(0)
    return json.loads(path.read_text(encoding="utf-8"))


def hit_in_retrieved(item: Dict[str, Any], retrieved: List[Dict]) -> List[bool]:
    """
    判断每个检索结果是否「命中」该问题的正确来源。

    命中规则：
        - 若提供了 expected_chunk_ids，则需 source 和 chunk_id 都对上（严格）；
        - 否则只要 source 文件名对上即算命中（宽松）。
    返回与 retrieved 等长的布尔列表，表示每个位次是否命中。
    """
    exp_sources = set(item.get("expected_sources", []))
    exp_chunks = set(item.get("expected_chunk_ids", []))
    flags = []
    for r in retrieved:
        if exp_chunks:
            flags.append(r["source"] in exp_sources and r["chunk_id"] in exp_chunks)
        else:
            flags.append(r["source"] in exp_sources)
    return flags


def evaluate_retrieval(golden: List[Dict], model, top_k: int) -> Dict[str, float]:
    """逐题检索并汇总 Hit Rate / Recall@K / MRR。"""
    n = len(golden)
    hit_count = 0
    recall_sum = 0.0
    rr_sum = 0.0  # reciprocal rank 之和

    print(f"\n{'='*60}\n检索层评测（Top-{top_k}，{n} 个问题）\n{'='*60}")

    for item in golden:
        retrieved = query_similar_chunks(
            query=item["question"], model=model, top_k=top_k,
            collection_name=CHROMA_COLLECTION,
        )
        flags = hit_in_retrieved(item, retrieved)

        # Hit Rate：是否至少命中一个
        hit = any(flags)
        hit_count += int(hit)

        # Recall@K：命中的正确来源数 / 应命中总数
        exp_chunks = item.get("expected_chunk_ids", [])
        if exp_chunks:
            denom = len(exp_chunks)
            matched = sum(1 for r in retrieved
                          if r["source"] in set(item["expected_sources"])
                          and r["chunk_id"] in set(exp_chunks))
        else:
            # 没给精确段号时，以「命中位次数」近似召回，denom 取实际命中（避免除零）
            denom = max(1, sum(flags))
            matched = sum(flags)
        recall_sum += matched / denom if denom else 0.0

        # MRR：第一个命中的位次倒数
        rr = 0.0
        for rank, f in enumerate(flags, start=1):
            if f:
                rr = 1.0 / rank
                break
        rr_sum += rr

        status = "✅" if hit else "❌"
        print(f"{status} rr={rr:.2f} | {item['question'][:40]}")

    return {
        "hit_rate": hit_count / n,
        "recall@k": recall_sum / n,
        "mrr": rr_sum / n,
    }


def judge_faithfulness(question: str, answer: str, context: str, client) -> float:
    """
    用 Claude 当裁判，判断 answer 是否仅依据 context（忠实度）。
    返回 0~1 分（1 = 完全有据，0 = 明显编造）。
    """
    prompt = (
        "你是严格的事实核查员。判断【答案】中的每个关键事实是否都能在【参考资料】中找到依据。\n"
        "评分标准（只输出一个 0 到 1 之间的小数，不要任何解释）：\n"
        "1.0 = 所有事实都有据；0.5 = 部分有据部分编造；0.0 = 主要靠编造。\n\n"
        f"【参考资料】\n{context}\n\n【问题】\n{question}\n\n【答案】\n{answer}\n\n"
        "忠实度分数："
    )
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    try:
        return max(0.0, min(1.0, float(raw.split()[0])))
    except (ValueError, IndexError):
        return -1.0  # 解析失败标记


def evaluate_generation(golden: List[Dict], model, top_k: int) -> Dict[str, float]:
    """对每题生成答案并评忠实度（需调用 Claude，有成本）。"""
    from utils.chat import chat_with_claude

    client = get_anthropic_client()
    n = len(golden)
    score_sum = 0.0
    valid = 0

    print(f"\n{'='*60}\n生成层评测 · Faithfulness（调用 Claude，{n} 个问题）\n{'='*60}")

    for item in golden:
        retrieved = query_similar_chunks(
            query=item["question"], model=model, top_k=top_k,
            collection_name=CHROMA_COLLECTION,
        )
        answer, _ = chat_with_claude(
            question=item["question"], retrieved_chunks=retrieved,
            conversation_history=[], model_name=CLAUDE_MODEL,
        )
        context = build_rag_context(retrieved)
        score = judge_faithfulness(item["question"], answer, context, client)
        if score >= 0:
            score_sum += score
            valid += 1
        print(f"忠实度={score:.2f} | {item['question'][:40]}")

    return {"faithfulness": score_sum / valid if valid else 0.0}


def main():
    parser = argparse.ArgumentParser(description="行研助手 RAG 离线评测")
    parser.add_argument("--judge", action="store_true",
                        help="额外评测生成忠实度（调用 Claude，需 API Key 且有成本）")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="检索段落数")
    args = parser.parse_args()

    golden = load_golden_set(GOLDEN_PATH)
    print(f"已加载黄金集：{len(golden)} 个问题")

    print("正在加载嵌入模型...")
    model = load_embedding_model(EMBEDDING_MODEL)

    retr = evaluate_retrieval(golden, model, args.top_k)

    print(f"\n{'─'*60}\n📊 检索层汇总")
    print(f"   Hit Rate  : {retr['hit_rate']:.3f}  （至少命中1段的问题比例）")
    print(f"   Recall@{args.top_k}  : {retr['recall@k']:.3f}  （正确来源被召回比例·最关键）")
    print(f"   MRR       : {retr['mrr']:.3f}  （首个命中越靠前越高）")

    if args.judge:
        gen = evaluate_generation(golden, model, args.top_k)
        print(f"\n📊 生成层汇总")
        print(f"   Faithfulness: {gen['faithfulness']:.3f}  （答案有据程度·防幻觉）")

    print(f"{'─'*60}\n✅ 评测完成。改参数后重跑，对比这几个数字即可。\n")


if __name__ == "__main__":
    main()
