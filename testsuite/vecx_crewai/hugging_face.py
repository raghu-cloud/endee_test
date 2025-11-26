from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbedder(EmbeddingFunction):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, documents):
        if isinstance(documents, str):
            documents = [documents]
        return self.model.encode(documents, convert_to_numpy=True)