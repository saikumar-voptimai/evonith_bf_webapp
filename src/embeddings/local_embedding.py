from sentence_transformers import SentenceTransformer
from utils.settings import settings

class LocalEmbeddingClient:
    """
    Local Embedding Client using Sentence Transformers (all-MiniLM-L6-v2 by default).
    Produces 384-dimensional embeddings as configured in the local settings.
    """
    def __init__(self):
        cfg = settings.embedding["local"]
        self.model_name = cfg.model_name
        self.device = cfg.device
        self.dimension = cfg.dimension
        
        # Load the model. Let it decide the device to avoid 'meta tensor' errors on some systems.
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str):
        """
        Produce a list of floats representing the embedding of the input text.
        """
        # convert result to list because qdrant_client expects it
        embedding = self.model.encode(text).tolist()
        return embedding