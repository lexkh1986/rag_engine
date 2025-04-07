# RAG Module

## Installation
1. Tạo virtual environment: `python -m venv venv`
2. Kích hoạt: `source venv/bin/activate` (Linux/Mac) hoặc `venv\Scripts\activate` (Windows)
3. Cài dependencies: `pip install -r requirements.txt`
4. Cấu hình `.env` với `CODEBASE_PATH` và `RAG_STORAGE_PATH`.

## Xây dựng RAG
Chạy: `python src/rag_builder.py`

## Kiểm tra RAG
Chạy: `python src/rag_loader.py`