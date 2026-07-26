"""
المرحلة 6: تجهيز الـ retriever اللي بيسترجع أقرب chunks لسؤال المستخدم.
نفس المنطق اللي كان في app.py قبل التقسيم (retriever_node) — من غير أي تغيير.
"""
from config import TOP_K


def get_retriever(vector_db, k=TOP_K):
    """بترجع retriever جاهز يسترجع أقرب k من الـ chunks لأي سؤال."""
    return vector_db.as_retriever(search_kwargs={"k": k})
