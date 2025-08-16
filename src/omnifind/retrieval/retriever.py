from ..embeddings.embedder import Embedder
from .faiss_index import SimpleIndex
from ..data.loader import load_sample_catalog
from ..logger import get_logger
import numpy as np

logger = get_logger('retriever')

class RetrieverService:
    def __init__(self, model_name=None):
        self.embedder = Embedder(model_name)
        self.index = SimpleIndex(dim=384)
        self.products = {}
        self._built = False

    def build_index_from_catalog(self, filename='sample_products.json'):
        data = load_sample_catalog(filename)
        if not data:
            return
        texts = [p.get('title','') + ' ' + p.get('brand','') for p in data]
        emb = self.embedder.encode_texts(texts)
        ids = [p['id'] for p in data]
        self.products = {p['id']: p for p in data}
        self.index.add(ids, emb)
        self.index.build()
        self._built = True
        logger.info('Index built with %d products' % len(ids))

    def search_text(self, query, top_k=5):
        if not self._built:
            self.build_index_from_catalog()
        vec = self.embedder.encode_texts([query])[0]
        ids, dists = self.index.search(vec, top_k=top_k)
        results = [self.products.get(i) for i in ids]
        return results

    def search_image(self, image_bytes, top_k=5):
        if not self._built:
            self.build_index_from_catalog()
        vec = self.embedder.encode_image(image_bytes)
        ids, dists = self.index.search(vec, top_k=top_k)
        results = [self.products.get(i) for i in ids]
        return results
