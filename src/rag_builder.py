# src/rag_builder.py
import os
import warnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from pathlib import Path
import pathspec

# Suppress FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Configuration
CODEBASE_PATH = os.getenv("CODEBASE_PATH")
RAG_STORAGE_PATH = os.getenv("RAG_STORAGE_PATH")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_ignore_patterns(base_path):
    """
    Load ignore patterns from .ragignore or .gitignore.
    Prioritizes .ragignore if it exists, otherwise falls back to .gitignore.

    Args:
        base_path (str): Base directory of the codebase.

    Returns:
        pathspec.PathSpec: Parsed ignore patterns, or None if no ignore file found.
    """
    ignore_file = Path(base_path) / ".ragignore"
    if not ignore_file.exists():
        ignore_file = Path(base_path) / ".gitignore"

    if ignore_file.exists():
        with open(ignore_file, "r", encoding="utf-8") as f:
            patterns = f.read().splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return None

def load_codebase(directory):
    """
    Load content from all files in the codebase directory, excluding ignored patterns.

    Args:
        directory (str): Path to the codebase directory.

    Returns:
        tuple: (list of file contents, list of file paths)
    """
    ignore_spec = load_ignore_patterns(directory)
    text_contents = []
    file_paths = []

    for root, dirs, files in os.walk(directory, topdown=True):
        rel_root = Path(root).relative_to(directory)  # Relative path of current directory

        # Skip entire directory if it matches ignore patterns
        if ignore_spec and ignore_spec.match_file(rel_root):
            dirs[:] = []  # Clear dirs to prevent further traversal
            continue

        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(directory)  # Relative path for ignore check

            # Skip if file matches ignore patterns
            if ignore_spec and ignore_spec.match_file(rel_path):
                # print(f"Ignored: {rel_path}")
                continue

            try:
                # Read file as text, skip if binary or unreadable
                with open(file_path, "r", encoding="utf-8") as f:
                    text_contents.append(f.read())
                file_paths.append(str(file_path))
            except (UnicodeDecodeError, IOError) as e:
                pass
                # print(f"Skipped {file_path}: {e} (likely binary or unreadable)")

    return text_contents, file_paths

def build_rag():
    """
    Build and save RAG embeddings from the codebase.
    Uses GPU if available for faster embedding generation.
    """
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Load model with error handling
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load codebase content
    texts, file_paths = load_codebase(CODEBASE_PATH)
    if not texts:
        print("No readable files found in codebase after filtering!")
        return

    # Generate embeddings
    try:
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # Save embeddings and metadata
    os.makedirs(RAG_STORAGE_PATH, exist_ok=True)
    np.save(f"{RAG_STORAGE_PATH}/embeddings.npy", embeddings)
    with open(f"{RAG_STORAGE_PATH}/file_paths.txt", "w") as f:
        f.write("\n".join(file_paths))
    print(f"RAG built and saved to {RAG_STORAGE_PATH}")

if __name__ == "__main__":
    build_rag()