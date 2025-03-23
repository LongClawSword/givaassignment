import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from data_loader import load_data
import pickle

df = load_data()
documents = df["text"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents, convert_to_numpy=True)

metric = "cosine" 

if metric == "cosine":
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  
elif metric == "dot":
    index = faiss.IndexFlatIP(embeddings.shape[1])  
else:
    index = faiss.IndexFlatL2(embeddings.shape[1])  

index.add(embeddings)

faiss.write_index(index, "vector.index")
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"FAISS index built using {metric} similarity.")
