"""
المرحلة 3: تقطيع المستندات لأجزاء (chunks) بحجم مناسب مع تداخل.
نفس المنطق اللي كان في دالة execute_vector_ingestion (Step 2) جوا app.py قبل التقسيم.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """بتقسم قائمة الـ Documents لأجزاء أصغر جاهزة لعمل embeddings ليها."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
