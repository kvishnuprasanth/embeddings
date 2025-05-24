from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

class EmbeddingRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
def embed_texts(req: EmbeddingRequest):
    embeddings = model.encode(req.texts, normalize_embeddings=True)
    return {"embeddings": embeddings.tolist()}
