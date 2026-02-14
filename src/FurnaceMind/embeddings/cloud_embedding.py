import io
import voyageai
from PIL import Image
from FurnaceMind.utils.settings import settings


class CloudEmbeddingClient:
    """
    Voyage Multimodal Embedding Client
    Supports text + image (+ video future ready)
    """

    def __init__(self):
        self.model = settings.embedding["cloud"].model_name
        self.dimension = settings.embedding["cloud"].dimension

        self.client = voyageai.Client(
            api_key=settings.embedding["cloud"].api_key
        )

    # ---------------------------------------
    # 🔹 Text Embedding
    # ---------------------------------------
    def embed_text(self, text: str):

        inputs = [[text]]

        result = self.client.multimodal_embed(
            inputs,
            model=self.model,
        )

        return result.embeddings[0]

    # ---------------------------------------
    # 🔹 Image Embedding
    # ---------------------------------------
    def embed_image(self, image_bytes: bytes):

        image = Image.open(io.BytesIO(image_bytes))

        inputs = [[image]]

        result = self.client.multimodal_embed(
            inputs,
            model=self.model,
        )

        return result.embeddings[0]