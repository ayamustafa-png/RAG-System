"""
المرحلة 1: تحميل ملفات PDF من مجلد papers_to_chat.
نفس المنطق اللي كان في دالة execute_vector_ingestion (Step 1) جوا app.py قبل التقسيم.
"""
from langchain_community.document_loaders import PyPDFDirectoryLoader

from config import PAPERS_DIR


def load_documents(papers_dir=PAPERS_DIR):
    """بترجع قائمة langchain Documents من كل ملفات الـ PDF في المجلد."""
    loader = PyPDFDirectoryLoader(papers_dir)
    return loader.load()
