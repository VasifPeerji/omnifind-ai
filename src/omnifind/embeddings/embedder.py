# Lightweight embedder abstraction.
# Uses sentence-transformers if available, otherwise falls back to a dummy embedder.

import numpy as np
from ..logger import get_logger

logger = get_logger('embedder')

try:
    from sentence_transformers import SentenceTransformer
    _has_sbert = True
except Exception:
    _has_sbert = False

class Embedder:
    def __init__(self, model_name=None):
        self.model_name = model_name
        if _has_sbert:
            model_name = model_name or 'sentence-transformers/all-MiniLM-L6-v2'
            logger.info(f'Loading SentenceTransformer: {model_name}')
            self.model = SentenceTransformer(model_name)
        else:
            logger.info('sentence-transformers not available, using dummy embedder')
            self.model = None

    def encode_texts(self, texts):
        if self.model:
            emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return emb
        # dummy embedding: hash-based vector
        out = []
        for t in texts:
            h = abs(hash(t)) % 1000
            vec = np.ones(384) * (h / 1000.0)
            out.append(vec.astype('float32'))
        return np.stack(out)

    def encode_image(self, image_bytes):
        # placeholder: for a real pipeline, load image and compute CLIP embedding
        logger.info('encode_image called: returning dummy vector')
        return np.ones(384, dtype='float32')
