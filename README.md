# RAG Module

## Installation
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Config `.env` with `CODEBASE_PATH` & `RAG_STORAGE_PATH`.

## Build RAG
Run: `python src/rag_builder.py`

## Check RAG
Run: `python src/rag_loader.py`