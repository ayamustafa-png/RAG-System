"""
إعدادات ومسارات مشتركة لكل ملفات المشروع.
نفس القيم بالظبط اللي كانت متحطة جوا app.py قبل التقسيم — مفيش قيمة اتغيرت.
"""

# --- مسارات النظام ---
PAPERS_DIR = "papers_to_chat"
DB_DIR = "chroma_db"

# --- فئات تصنيف الأبحاث (الموديل اللي اتدرب في train_model.py) ---
TOP_CATEGORIES = ["cs", "math", "physics", "astro-ph"]
MODEL_PATH = "academic_classifier_model.h5"
TOKENIZER_PATH = "tokenizer.json"
MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 15000
EMBEDDING_DIM = 64

# --- إعدادات التقطيع (Chunking) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- إعدادات الـ Embeddings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- إعدادات الـ LLM (التوليد) ---
LLM_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
LLM_TASK = "conversational"
LLM_PROVIDER = "auto"
LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 512

# --- إعدادات الاسترجاع ---
TOP_K = 4

# --- الـ system prompt المستخدم في التوليد ---
SYSTEM_PROMPT = (
    "You are a highly analytical academic research assistant. Formulate an authoritative, objective reply "
    "based strictly on the provided context. Maintain academic integrity. If the answer cannot be confidently "
    "inferred from the retrieved data, explicitly respond with: 'The requested information is not available "
    "within the ingested references.' and do not extrapolate or hallucinate.\n\n"
    "Retrieved References Context:\n{context}"
)
