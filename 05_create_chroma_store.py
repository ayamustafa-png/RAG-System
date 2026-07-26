"""
المرحلة 5: إنشاء قاعدة Chroma من الـ chunks (وقت الفهرسة)، أو تحميلها لو كانت
موجودة أصلاً على الديسك (وقت التشغيل العادي).
نفس المنطق اللي كان في app.py قبل التقسيم — من غير أي تغيير.
"""
from langchain_chroma import Chroma

from config import DB_DIR


def create_chroma_store(chunks, embedding_model, persist_directory=DB_DIR):
    """بتبني قاعدة Chroma جديدة من الـ chunks وتحفظها على الديسك."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )


def load_chroma_store(embedding_model, persist_directory=DB_DIR):
    """بتحمّل قاعدة Chroma موجودة أصلاً من الديسك (من غير إعادة بناء)."""
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
