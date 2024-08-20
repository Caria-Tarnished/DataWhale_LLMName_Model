from sentence_transformers import SentenceTransformer
import numpy as np

# 加载预训练的句子嵌入模型
model = SentenceTransformer('AI-ModelScope/bge-small-zh-v1.5')

# 读取文档库
with open('poetry_corpus.txt', 'r', encoding='utf-8') as f:
    documents = f.read().splitlines()

# 向量化文档库中的每个段落
embeddings = model.encode(documents)

# 保存嵌入向量
np.save('poetry_embeddings.npy', embeddings)

print("文档向量化完成，嵌入向量已保存至 poetry_embeddings.npy")
