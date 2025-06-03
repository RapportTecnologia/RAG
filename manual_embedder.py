from sentence_transformers import SentenceTransformer

class ManualEmbedder:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, documents):
        # Espera uma lista de strings (conteúdo dos documentos)
        return self.model.encode(documents, show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode([query])[0]
