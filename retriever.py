import faiss
bm25_path: str = "artifacts/bm25_tokens.pkl",
embedding_model_name: str = "all-MiniLM-L6-v2"):
self.index = faiss.read_index(index_path)
with open(docs_path, "r", encoding="utf-8") as f:
self.docs = json.load(f)
self.embeddings = np.load(embeddings_path)
with open(bm25_path, "rb") as f:
bm25_meta = pickle.load(f)
self.bm25_tokens = bm25_meta["tokens"]
self.bm25 = BM25Okapi(self.bm25_tokens)
self.model = SentenceTransformer(embedding_model_name)


def _embed(self, texts: List[str]):
emb = self.model.encode(texts, convert_to_numpy=True)
faiss.normalize_L2(emb)
return emb


def retrieve(self, query: str, top_k: int = 8, alpha: float = 0.5) -> List[Dict]:
# embed query
q_emb = self._embed([query])
D, I = self.index.search(q_emb, top_k)
faiss_scores = D[0].tolist()
doc_ids = I[0].tolist()


# BM25 scores across corpus
q_tokens = query.split()
bm25_scores = self.bm25.get_scores(q_tokens)


# build combined score list for returned doc ids
results = []
# normalize bm25 and faiss for combination
bm25_vals = [bm25_scores[i] for i in doc_ids]
# simple normalization
bm25_arr = np.array(bm25_vals, dtype=float)
if bm25_arr.max() - bm25_arr.min() > 0:
bm25_norm = (bm25_arr - bm25_arr.min()) / (bm25_arr.max() - bm25_arr.min())
else:
bm25_norm = np.zeros_like(bm25_arr)
faiss_arr = np.array(faiss_scores, dtype=float)
# faiss scores are cosine in [-1,1], normalize to 0..1
faiss_norm = (faiss_arr + 1.0) / 2.0
if faiss_norm.max() - faiss_norm.min() > 0:
faiss_norm = (faiss_norm - faiss_norm.min()) / (faiss_norm.max() - faiss_norm.min())


for idx, doc_id in enumerate(doc_ids):
doc = self.docs[doc_id]
combined = alpha * faiss_norm[idx] + (1 - alpha) * bm25_norm[idx]
results.append({
"id": doc_id,
"text": doc["text"],
"source": doc.get("source", "HR-Policy.pdf"),
"faiss_score": float(faiss_scores[idx]),
"bm25_score": float(bm25_vals[idx]),
"combined_score": float(combined)
})
results = sorted(results, key=lambda r: r['combined_score'], reverse=True)
return results
