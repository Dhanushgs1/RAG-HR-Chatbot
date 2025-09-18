import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from rank_bm25 import BM25Okapi




def build(artifact_docs_path: str = "artifacts/docs.json",
embedding_model_name: str = "all-MiniLM-L6-v2",
out_index_path: str = "artifacts/faiss.index",
out_embeddings_path: str = "artifacts/embeddings.npy",
out_bm25_path: str = "artifacts/bm25_tokens.pkl"):
os.makedirs("artifacts", exist_ok=True)
with open(artifact_docs_path, "r", encoding="utf-8") as f:
docs = json.load(f)


texts = [d['text'] for d in docs]
model = SentenceTransformer(embedding_model_name)
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


# normalize
faiss.normalize_L2(embeddings)


d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)


faiss.write_index(index, out_index_path)
np.save(out_embeddings_path, embeddings)


# prepare BM25 tokens for optional re-ranking
tokenized = [t.split() for t in texts]
bm25 = BM25Okapi(tokenized)
with open(out_bm25_path, "wb") as f:
pickle.dump({"tokens": tokenized}, f)


print("Index built and saved.")




if __name__ == '__main__':
build()
