# Simple script to compute embeddings for the sample catalog and persist them.
from src.omnifind.embeddings.embedder import Embedder
from src.omnifind.data.loader import load_sample_catalog
import numpy as np
import json
from pathlib import Path

def run():
    data = load_sample_catalog()
    if not data:
        print('No data found.')
        return
    texts = [p.get('title','') + ' ' + p.get('brand','') for p in data]
    emb = Embedder().encode_texts(texts)
    out = {'ids':[p['id'] for p in data], 'embeddings': emb.tolist()}
    p = Path('data') / 'embeddings.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        json.dump(out, f)
    print('Saved embeddings to', p)

if __name__ == '__main__':
    run()
