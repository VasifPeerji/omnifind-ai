import pickle
from pathlib import Path
import numpy as np
import faiss

EMBED_FILE = Path("data/embeddings/text_embeddings.pkl")
INDEX_FILE = Path("data/embeddings/faiss_index.index")

def main():
    if not EMBED_FILE.exists():
        raise FileNotFoundError(f"Embedding file not found: {EMBED_FILE}")

    with open(EMBED_FILE, "rb") as f:
        data = pickle.load(f)

    embeddings = np.array(data["embeddings"]).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Saved FAISS index → {INDEX_FILE} ({index.ntotal} vectors)")

if __name__ == "__main__":
    main()
