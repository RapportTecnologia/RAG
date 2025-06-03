import os
import pickle
import faiss
import numpy as np
from manual_embedder import ManualEmbedder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
INDEX_FILE = os.path.join(VECTORSTORE_DIR, "faiss.index")
DOCS_FILE = os.path.join(VECTORSTORE_DIR, "docs.pkl")
META_FILE = os.path.join(VECTORSTORE_DIR, "meta.pkl")


def get_documents_metadata(documents):
    """Retorna lista de (nome, mtime) dos arquivos usados nos documentos."""
    meta = []
    for doc in documents:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            try:
                fname = doc.metadata['source']
                mtime = os.path.getmtime(fname)
                meta.append((fname, mtime))
            except Exception:
                pass
    return sorted(meta)


def save_vectorstore(index, docs, meta):
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, 'wb') as f:
        pickle.dump(docs, f)
    with open(META_FILE, 'wb') as f:
        pickle.dump(meta, f)


def load_vectorstore():
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, 'rb') as f:
        docs = pickle.load(f)
    with open(META_FILE, 'rb') as f:
        meta = pickle.load(f)
    return index, docs, meta


def vectorstore_exists():
    return os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE) and os.path.exists(META_FILE)
