"""
demo_embedding.py —— 一次性教学脚本
目的：亲眼看看 Sentence Transformer 怎么把句子变成向量，以及余弦相似度怎么算。
用的就是 app.py 里同一个模型 paraphrase-multilingual-MiniLM-L12-v2。
看完即可删除。
"""

import sys
from sentence_transformers import SentenceTransformer
import numpy as np

# Windows 控制台默认 GBK，强制用 UTF-8 输出，避免中文乱码
sys.stdout.reconfigure(encoding="utf-8")

# 1) 加载和你项目里完全相同的嵌入模型
print("正在加载模型（本地缓存，应该很快）...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2) 准备 4 句话：前两句意思相近但字面不同；第三句话题完全不同；第四句是英文同义句
sentences = [
    "半导体行业前景广阔",      # A
    "芯片产业未来很有潜力",    # B —— 和 A 意思像，但没一个字相同
    "今天午饭吃什么呢",        # C —— 完全不相关
    "The semiconductor industry has a promising future",  # D —— A 的英文版
]

# 3) 向量化：每句话 → 一串数字（向量）
vectors = model.encode(sentences)
print(f"\n每个句子被编码成一个 {vectors.shape[1]} 维向量")
print(f"例如「{sentences[0]}」的向量前 8 个数字：")
print(np.round(vectors[0][:8], 4), "...（后面还有 376 个）")


# 4) 余弦相似度：衡量两个向量"方向"有多接近
#    公式：cos = (a·b) / (|a| * |b|)，范围 [-1, 1]，越接近 1 越相似
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n" + "=" * 50)
print("余弦相似度（越接近 1 越相似）：")
print("=" * 50)

pairs = [
    (0, 1, "A-B  中文近义（字面不同）"),
    (0, 2, "A-C  完全不相关"),
    (0, 3, "A-D  中文 vs 英文同义"),
    (1, 2, "B-C  完全不相关"),
]
for i, j, label in pairs:
    sim = cosine(vectors[i], vectors[j])
    bar = "#" * int(sim * 30) if sim > 0 else ""
    print(f"{label:28s} {sim:+.3f}  {bar}")

print("\n看点：")
print("  - A↔B 明显高于 A↔C —— 模型抓的是『意思』不是『字面』")
print("  - A↔D 也偏高 —— 多语言模型让中英文同义句向量也靠近")
print("\n这正是 RAG 检索的原理：把『问题』也编码成向量，")
print("再找库里余弦相似度最高的 top_k 个块。")
