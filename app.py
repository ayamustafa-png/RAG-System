import sys

# --- 0. SQLite3 Workaround (required on Streamlit Community Cloud for ChromaDB) ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import json
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. System Environment Architecture ---
PAPERS_DIR = "papers_to_chat"
DB_DIR = "chroma_db"
TOP_CATEGORIES = ["cs", "math", "physics", "astro-ph"]
MODEL_PATH = "academic_classifier_model.h5"
TOKENIZER_PATH = "tokenizer.json"
MAX_SEQUENCE_LENGTH = 200

if not os.path.exists(PAPERS_DIR):
    os.makedirs(PAPERS_DIR)

st.set_page_config(
    page_title="Smart Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================== CSS مخصص ============================== #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Cairo', sans-serif;
    }

    .main-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 22px 28px;
        border-radius: 18px;
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        box-shadow: 0 8px 30px rgba(37, 99, 235, 0.25);
        margin-bottom: 20px;
    }
    .main-header .icon-badge {
        font-size: 38px;
        background: rgba(255,255,255,0.18);
        border-radius: 14px;
        padding: 8px 14px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 24px;
        font-weight: 800;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 4px 0 0 0;
        font-size: 14px;
    }

    .metric-card {
        background: #f8f9fc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #2563EB;
    }
    .metric-card .metric-label {
        font-size: 12px;
        color: #6b7280;
        margin-top: 2px;
    }

    .source-card {
        background: #f8f9fc;
        border-right: 3px solid #7C3AED;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .source-card .source-title {
        color: #7C3AED;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }
    .status-ok { background: rgba(16, 185, 129, 0.12); color: #059669; }
    .status-bad { background: rgba(239, 68, 68, 0.12); color: #dc2626; }

    .file-chip {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 12px;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================== الهيدر ============================== #
st.markdown(
    """
    <div class="main-header">
        <div class="icon-badge">📚</div>
        <div>
            <h1>Smart Academic Research Assistant</h1>
            <p>Hybrid RAG + Deep Learning Framework — تصنيف ذكي للأبحاث + إجابة بالاعتماد على مستنداتك</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- 2. Cached Resource Initialization ---
VOCAB_SIZE = 15000
EMBEDDING_DIM = 64


def _build_classifier_architecture():
    # Must match train_model.py EXACTLY (same layer order/types) so that
    # load_weights() below can map the saved weights correctly. We rebuild
    # the architecture instead of using tf.keras.models.load_model() because
    # the .h5 file can be produced by a slightly different Keras version than
    # what's installed on Streamlit Cloud, which makes full-config loading
    # (load_model) fail with a TypeError on from_config. Loading weights only
    # sidesteps that config-schema mismatch.
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,)),
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(TOP_CATEGORIES), activation="softmax"),
        ]
    )


@st.cache_resource(show_spinner="Loading models and services...")
def initialize_system_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(base_dir, MODEL_PATH)
    tokenizer_file = os.path.join(base_dir, TOKENIZER_PATH)

    dl_model = _build_classifier_architecture()
    dl_model.load_weights(model_file)

    with open(tokenizer_file, "r", encoding="utf-8") as handle:
        token_generator = tokenizer_from_json(handle.read())

    embedding_client = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    hf_token = st.secrets.get("HF_TOKEN")
    if not hf_token:
        st.error(
            "HF_TOKEN مش موجود في الـ Secrets بتاعة التطبيق.\n\n"
            "روحي Streamlit Cloud → App settings → Secrets وضيفي سطر زي كده:\n\n"
            'HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"'
        )
        st.stop()

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="conversational",  # this model is only served as a chat model on the "together" provider
        provider="auto",  # let Hugging Face route to whichever partner currently serves this model
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token,
    )
    llm_node = ChatHuggingFace(llm=llm_endpoint)

    return dl_model, token_generator, embedding_client, llm_node


if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    dl_model, tokenizer, embeddings, llm = initialize_system_resources()
else:
    st.error(
        f"الملفين '{MODEL_PATH}' أو '{TOKENIZER_PATH}' مش موجودين في الريبو. "
        "شغّلي train_model.py الأول عندك لوكال، وارفعي الملفين دول (h5 + pickle) على GitHub "
        "جنب app.py."
    )
    st.stop()


# --- 3. ETL Document Embedding Ingestion Pipeline ---
def execute_vector_ingestion():
    with st.spinner("Executing document ETL processing, text-splitting, and vector persistence..."):
        st.write("Step 1: Loading PDFs...")
        loader = PyPDFDirectoryLoader(PAPERS_DIR)
        documents = loader.load()
        st.write(f"Loaded {len(documents)} documents.")

        if not documents:
            st.sidebar.error(f"Ingestion directory '{PAPERS_DIR}' contains zero documents.")
            return None

        st.write("Step 2: Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_documents(documents)
        st.write(f"Created {len(chunks)} chunks.")

        st.write("Step 3: Creating embeddings and storing in ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR,
        )
        st.write("Step 4: Finished.")

        st.sidebar.success(f"ETL Execution complete. Persisted {len(chunks)} chunks.")
        return vector_store


# --- 4. Sidebar: File Upload + Control Panel ---
def get_indexed_pdf_files():
    if not os.path.exists(PAPERS_DIR):
        return []
    return sorted(f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf"))


with st.sidebar:
    st.markdown("### 📤 رفع مستندات جديدة")

    uploaded_files = st.file_uploader(
        "ارفعي ملف أو أكتر بصيغة PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        new_files_saved = 0
        for uploaded_file in uploaded_files:
            destination_path = os.path.join(PAPERS_DIR, uploaded_file.name)
            # منتكتبش الملف تاني لو موجود أصلاً بنفس الاسم ومحفوظ قبل كده
            if not os.path.exists(destination_path):
                with open(destination_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_saved += 1

        if new_files_saved:
            st.success(f"✅ اتحفظ {new_files_saved} ملف جديد في '{PAPERS_DIR}/'.")

    current_files = get_indexed_pdf_files()
    if current_files:
        st.caption(f"📄 {len(current_files)} ملف جاهز للفهرسة:")
        chips_html = "".join(f'<span class="file-chip">{f}</span>' for f in current_files)
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.caption("مفيش ملفات لسه في المجلد.")

    st.markdown("---")
    st.header("⚙️ System Control Panel")
    st.markdown(
        f"**Instructions:**\n1. ارفعي ملفات PDF من فوق (أو حطيهم يدوي في `{PAPERS_DIR}/`).\n"
        "2. شغّلي الـ ingestion pipeline تحت."
    )

    if st.button("🔄 Execute Ingestion Pipeline", use_container_width=True):
        execute_vector_ingestion()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Infrastructure Specifications:")
    st.info(
        "• DL Intent Engine: **LSTM (Keras Backend)**\n"
        "• Vector DB Hub: **ChromaDB Target**\n"
        "• Embedding Model: **All-MiniLM-L6-v2 (HuggingFace)**\n"
        "• Core Generative LLM: **Qwen2.5-7B-Instruct (HuggingFace Inference Providers)**"
    )

    st.markdown("---")
    if st.button("🗑️ مسح سجل المحادثة", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Initialize Vector DB link if data exists
vector_db = None
if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# ============================== لوحة الإحصائيات ============================== #
stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(get_indexed_pdf_files())}</div>
            <div class="metric-label">📄 ملفات مرفوعة</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col2:
    chunk_count = "—"
    if vector_db is not None:
        try:
            chunk_count = vector_db._collection.count()
        except Exception:
            chunk_count = "—"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{chunk_count}</div>
            <div class="metric-label">🧩 Chunks مفهرسة</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col3:
    db_status_html = (
        '<span class="status-pill status-ok">🟢 قاعدة البيانات جاهزة</span>'
        if vector_db is not None
        else '<span class="status-pill status-bad">🔴 لسه مفيش فهرسة</span>'
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="margin-top:8px;">{db_status_html}</div>
            <div class="metric-label" style="margin-top:8px;">حالة الـ Vector DB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================== عرض سجل المحادثة ============================== #
for entry in st.session_state.chat_history:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(entry["query"])

    with st.chat_message("assistant", avatar="📚"):
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Predicted Academic Domain (DL Inference)", value=entry["predicted_class"].upper())
        with metric_col2:
            st.metric(label="Classifier Confidence Level", value=f"{entry['confidence']:.2f}%")

        if entry.get("answer"):
            st.write(entry["answer"])
            if entry.get("sources"):
                with st.expander(f"📄 المصادر المستخدمة ({len(entry['sources'])})"):
                    for i, src in enumerate(entry["sources"]):
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-title">📄 {src['file']} — صفحة {src['page']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.info(src["content"])
        elif entry.get("warning"):
            st.warning(entry["warning"])
        elif entry.get("error"):
            st.error(entry["error"])

# --- 5. Query Processing & Multi-Model Inference Logic ---
user_query = st.chat_input("اسألي سؤال بحثي أو الصقي abstract هنا...")

if user_query:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    # Execution Path A: Deep Learning Structural Intent Classification
    sequence = tokenizer.texts_to_sequences([user_query])
    padded_sequence = pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )

    prediction = dl_model.predict(padded_sequence)
    predicted_class = TOP_CATEGORIES[np.argmax(prediction)]
    confidence_score = np.max(prediction) * 100

    entry = {
        "query": user_query,
        "predicted_class": predicted_class,
        "confidence": confidence_score,
    }

    with st.chat_message("assistant", avatar="📚"):
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Predicted Academic Domain (DL Inference)", value=predicted_class.upper())
        with metric_col2:
            st.metric(label="Classifier Confidence Level", value=f"{confidence_score:.2f}%")

        # Execution Path B: Vector Search Retrieval and Contextual Generation
        if vector_db:
            retriever_node = vector_db.as_retriever(search_kwargs={"k": 4})

            system_instructions = (
                "You are a highly analytical academic research assistant. Formulate an authoritative, objective reply "
                "based strictly on the provided context. Maintain academic integrity. If the answer cannot be confidently "
                "inferred from the retrieved data, explicitly respond with: 'The requested information is not available "
                "within the ingested references.' and do not extrapolate or hallucinate.\n\n"
                "Retrieved References Context:\n{context}"
            )
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_instructions),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_orchestration_chain = create_retrieval_chain(retriever_node, question_answer_chain)

            with st.spinner("Retrieving local semantic knowledge and generating structural response..."):
                try:
                    start_time = time.time()
                    execution_response = rag_orchestration_chain.invoke({"input": user_query})
                    elapsed = time.time() - start_time

                    answer_text = execution_response["answer"]
                    st.write(answer_text)
                    st.caption(f"⏱️ {elapsed:.2f} ثانية")

                    sources = []
                    for document in execution_response["context"]:
                        sources.append({
                            "file": os.path.basename(document.metadata.get("source", "Unknown_Reference.pdf")),
                            "page": document.metadata.get("page", "N/A"),
                            "content": document.page_content,
                        })

                    if sources:
                        with st.expander(f"📄 المصادر المستخدمة ({len(sources)})"):
                            for src in sources:
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-title">📄 {src['file']} — صفحة {src['page']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.info(src["content"])

                    entry["answer"] = answer_text
                    entry["sources"] = sources
                except Exception as e:
                    error_message = (
                        f"حصل خطأ أثناء توليد الإجابة من الـ LLM: {e}\n\n"
                        "لو الرسالة بتقول إن الموديل مش متاح (not supported / 404)، غيّري "
                        "قيمة `repo_id` في app.py لموديل تاني متاح دلوقتي على "
                        "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation"
                    )
                    st.error(error_message)
                    entry["error"] = error_message
        else:
            warning_message = (
                "System Notice: RAG pipeline is offline. ارفعي ملفات PDF من الشريط الجانبي وشغّلي "
                "الـ Ingestion Pipeline الأول."
            )
            st.warning(warning_message)
            entry["warning"] = warning_message

    st.session_state.chat_history.append(entry)
