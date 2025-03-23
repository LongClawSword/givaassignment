# 🚀 Document Similarity Search API

This project provides an API to find similar documents using **vector embeddings** and **FAISS** for efficient search.

## 📌 Features
✅ Supports **multiple similarity metrics** (Cosine Similarity, L2 Distance, Dot Product).  
✅ Implements **real-time indexing** for newly added documents.  
✅ Uses **Hugging Face Sentence Transformers** for embedding generation.  
✅ FastAPI-based backend for high performance.  

---

## 🛠 Tech Stack
- **Backend**: FastAPI (Python)
- **Machine Learning**: Sentence Transformers (Hugging Face)
- **Vector Database**: FAISS
- **Deployment**: Railway / Render / Fly.io

---

## 🚀 Setup Instructions

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/yourusername/document-similarity-api.git
cd document-similarity-api
```

### **2️⃣ Install Dependencies**
```sh
pip install fastapi uvicorn faiss-cpu sentence-transformers pandas
```

### **3️⃣ Generate Embeddings**
```sh
python embedder.py
```

### **4️⃣ Run the API**
```sh
uvicorn main:app --reload
```

## 🔍 API Endpoints

### **1️⃣ Search for Similar Documents**
```sh
curl "http://127.0.0.1:8000/api/search?q=AI%20technology&metric=cosine"
```

### **2️⃣ Add a New Document**
```sh
curl -X POST "http://127.0.0.1:8000/api/add" -H "Content-Type: application/json" -d '{"text": "New AI model beats human performance"}'
```

## 👨‍💻 Author
Developed by Aryan Singh
📧 Contact: aryansingh7574@gmail.com

