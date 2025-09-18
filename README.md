# RAG HR Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) Chatbot** for answering HR policy questions based on the provided `HR-Policy.pdf` document. It uses FAISS, BM25, and Groq LLM for retrieval and answer generation, with a FastAPI backend and Streamlit frontend.

---

## 🚀 Features

* Extract and chunk HR policy PDF into searchable text segments.
* Generate embeddings using **SentenceTransformers** and store them in **FAISS**.
* Re-rank results using **BM25** + cosine scoring.
* Add query caching (SQLite) to avoid repeated LLM calls.
* Expose `/query` endpoint via **FastAPI**.
* Minimal **Streamlit UI** for chat interaction.
* Fully containerized with **Docker**.

---

## 📂 Repo Structure

```
rag-hr-chatbot/
├── README.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── start.sh
├── src/
│   ├── ingest.py
│   ├── build_index.py
│   ├── retriever.py
│   ├── cache.py
│   ├── llm_client.py
│   ├── api.py
│   └── streamlit_app.py
└── artifacts/
    ├── faiss.index
    ├── docs.json
    ├── embeddings.npy
    └── bm25_tokens.pkl
```

---

## ⚙️ Installation & Setup

### 1. Clone repo and setup environment

```bash
git clone https://github.com/<your-username>/rag-hr-chatbot.git
cd rag-hr-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and update with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
CACHE_TTL_SECONDS=86400
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Ingest HR Policy and build index

```bash
python src/ingest.py HR-Policy.pdf
python src/build_index.py
```

### 4. Run backend (FastAPI)

```bash
uvicorn src.api:app --reload --port 8000
```

### 5. Run frontend (Streamlit)

```bash
streamlit run src/streamlit_app.py --server.port 8501
```

Access at:

* API: [http://localhost:8000/docs](http://localhost:8000/docs)
* UI: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker

### Build image

```bash
docker build -t your-dockerhub-username/rag-hr-chatbot:latest .
```

### Run container

```bash
docker run -p 8000:8000 -p 8501:8501 your-dockerhub-username/rag-hr-chatbot:latest
```

### Push to Docker Hub

```bash
docker login
docker push your-dockerhub-username/rag-hr-chatbot:latest
```

---

## 🔐 Security Notes

* Keep `GROQ_API_KEY` in `.env`, never commit it.
* Sanitize any external PDFs before ingestion.
* Limit LLM response length and rate-limit queries.

---

## 📜 License

MIT License.

---

## ✨ Future Improvements

* Add Redis instead of SQLite for caching.
* Improve frontend design with chat history.
* Deploy with Kubernetes for scalability.
