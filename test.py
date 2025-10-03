import faiss
print(faiss.get_num_gpus())

import torch
print(torch.cuda.is_available())
