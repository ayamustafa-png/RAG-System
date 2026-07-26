"""
المرحلة 4: تجهيز نموذج تحويل النصوص لـ vectors (embeddings).
نفس المنطق اللي كان جوا initialize_system_resources في app.py قبل التقسيم
(نفس الموديل all-MiniLM-L6-v2 بالظبط).
"""
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME


def get_embedding_model():
    """بترجع كائن الـ embeddings المستخدم في كل من الفهرسة والاسترجاع."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
