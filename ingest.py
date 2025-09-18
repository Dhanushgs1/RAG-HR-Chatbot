import pdfplumber
import json
from typing import List




def load_pdf_text(path: str) -> str:
"""Extract text from PDF using pdfplumber."""
texts = []
with pdfplumber.open(path) as pdf:
for page in pdf.pages:
text = page.extract_text()
if text:
texts.append(text)
return "\n\n".join(texts)




def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
"""Chunk text into overlapping chunks (character-based)."""
chunks = []
start = 0
text_length = len(text)
while start < text_length:
end = start + chunk_size
chunk = text[start:end]
chunks.append(chunk.strip())
start = end - overlap
return [c for c in chunks if len(c) > 50]




if __name__ == "__main__":
import sys
if len(sys.argv) < 2:
print("usage: python ingest.py path/to/HR-Policy.pdf")
raise SystemExit(1)


in_pdf = sys.argv[1]
text = load_pdf_text(in_pdf)
chunks = chunk_text(text)
out_file = "artifacts/docs.json"
with open(out_file, "w", encoding="utf-8") as f:
docs = [{"id": i, "text": chunks[i], "source": in_pdf} for i in range(len(chunks))]
json.dump(docs, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(chunks)} chunks to {out_file}")