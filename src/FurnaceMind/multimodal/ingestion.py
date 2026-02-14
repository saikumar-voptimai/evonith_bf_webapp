import os
import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from FurnaceMind.multimodal.parsers import (
    parse_pdf,
    parse_docx,
    parse_pptx,
    parse_excel,
)


# ---------------------------------------------
# 🔹 Text Chunking
# ---------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# ---------------------------------------------
# 🔹 Main File Processor
# ---------------------------------------------
def process_file(file, knowledge_store, embedding_client):

    file_type = file.name.split(".")[-1].lower()

    # ==================================================
    # 🖼 IMAGE HANDLING (Multimodal Embedding)
    # ==================================================
    if file_type in ["png", "jpg", "jpeg"]:

        image_bytes = file.read()

        # Save image locally (so we can display later)
        upload_dir = Path("uploaded_images")
        upload_dir.mkdir(exist_ok=True)

        image_path = upload_dir / f"{uuid.uuid4()}_{file.name}"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Generate embedding
        embedding = embedding_client.embed_image(image_bytes)

        # Insert into Qdrant
        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": file.name,
                        "type": "image",
                        "file_path": str(image_path),
                    },
                )
            ],
            wait=True,
        )

        return


    # ==================================================
    # 📄 TEXT DOCUMENT HANDLING
    # ==================================================

    if file_type == "pdf":
        text = parse_pdf(file)

    elif file_type == "docx":
        text = parse_docx(file)

    elif file_type == "pptx":
        text = parse_pptx(file)

    elif file_type in ["xls", "xlsx"]:
        text = parse_excel(file)

    else:
        # Unsupported file type
        return


    if not text:
        return

    chunks = chunk_text(text)

    for chunk in chunks:
        knowledge_store.add_document(
            content=chunk,
            metadata={
                "source": file.name,
                "type": file_type,
            },
        )