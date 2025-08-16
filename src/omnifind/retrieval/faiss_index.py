# Simple FAISS index helper with save/load functionality.
# Uses faiss-cpu if available, otherwise acts as a simple in-memory linear scan.

import numpy as np
from ..logger import get_logger

logger = get_logger('faiss_index')

try:
    import faiss
    _has_faiss = True
except Exception:
    _has_faiss = False

class SimpleIndex:
    def __init__(self, dim=384):
        self.dim = dim
        self.vectors = []
        self.ids = []

    def add(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.ids.append(i)
            self.vectors.append(v.astype('float32'))

    def build(self):
        if _has_faiss:
            xb = np.stack(self.vectors)
            index = faiss.IndexFlatL2(self.dim)
            index.add(xb)
            self._index = index
            logger.info('FAISS index built')
        else:
            self._index = None
            logger.info('Using fallback linear index')

    def search(self, qvec, top_k=5):
        if _has_faiss and self._index is not None:
            D, I = self._index.search(qvec.reshape(1, -1).astype('float32'), top_k)
            ids = [self.ids[i] for i in I[0].tolist()]
            return ids, D[0].tolist()
        # fallback: linear scan
        dists = []
        q = qvec.astype('float32')
        for v in self.vectors:
            dists.append(np.sum((v - q)**2))
        idxs = np.argsort(dists)[:top_k]
        ids = [self.ids[i] for i in idxs]
        return ids, [dists[i] for i in idxs]
