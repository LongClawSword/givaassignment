from fastapi import FastAPI, Query
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

app = FastAPI()

with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

index_l2 = faiss.read_index("vector.index")  
index_cosine = faiss.read_index("vector.index")  
index_dot = faiss.read_index("vector.index") 

class DocumentInput(BaseModel):
    text: str

@app.get("/api/search")
def search(q: str, metric: str = Query("cosine", enum=["cosine", "l2", "dot"])):
    query_embedding = model.encode([q], convert_to_numpy=True)

    if metric == "cosine":
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        index = index_cosine
    elif metric == "dot":
        index = index_dot
    else:
        index = index_l2

    D, I = index.search(np.array(query_embedding), k=5)

    results = [{"id": int(i), "text": documents[i], "score": float(D[0][j])} for j, i in enumerate(I[0])]

    return {"query": q, "metric": metric, "results": results}

@app.post("/api/add")
def add_document(doc: DocumentInput):
    global documents, index_l2, index_cosine, index_dot

    documents.append(doc.text)

    new_embedding = model.encode([doc.text], convert_to_numpy=True)
    new_embedding_norm = new_embedding / np.linalg.norm(new_embedding)

    index_l2.add(new_embedding)
    index_cosine.add(new_embedding_norm)
    index_dot.add(new_embedding)

    faiss.write_index(index_l2, "vector.index")

    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    return {"message": "Document added successfully", "id": len(documents) - 1, "text": doc.text}