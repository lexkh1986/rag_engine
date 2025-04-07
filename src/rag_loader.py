# src/rag_loader.py
import os
import warnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Suppress FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Configuration
RAG_STORAGE_PATH = os.getenv("RAG_STORAGE_PATH")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class RAGLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME).to(self.device)
        self.embeddings = np.load(f"{RAG_STORAGE_PATH}/embeddings.npy")
        with open(f"{RAG_STORAGE_PATH}/file_paths.txt", "r") as f:
            self.file_paths = f.read().splitlines()

    def search(self, query, top_k=3):
        """Search for relevant documents based on query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.file_paths[i], scores[i]) for i in top_indices]
        return results

if __name__ == "__main__":
    rag = RAGLoader()
    query = "A FastAPI-based application for managing user transactions"
    results = rag.search(query)
    for file_path, score in results:
        print(f"File: {file_path}, Score: {score}")