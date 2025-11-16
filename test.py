# import faiss
# print(faiss.get_num_gpus())

# import torch
# print(torch.cuda.is_available())

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/clip-ViT-L-14")
print(m)
