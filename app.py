import sys

# --- 0. SQLite3 Workaround (required on Streamlit Community Cloud for ChromaDB) ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import time
import importlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from config import (
    PAPERS_DIR,
    DB_DIR,
    TOP_CATEGORIES,
    MODEL_PATH,
    TOKENIZER_PATH,
    MAX_SEQUENCE_LENGTH,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# --- Pipeline stage modules (filenames start with a digit, so importlib is required) ---
documents_stage = importlib.import_module("01_documents")
preprocessing_stage = importlib.import_module("02_preprocessing")
chunking_stage = importlib.import_module("03_chunking")
vector_stage = importlib.import_module("04_vector_representation")
chroma_stage = importlib.import_module("05_create_chroma_store")
retrieval_stage = importlib.import_module("06_retrieve_context")
prompting_stage = importlib.import_module("07_prompting")

CATEGORY_LABELS = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "astro-ph": "Astrophysics",
}
NOT_AVAILABLE_PHRASE = "not available"

# ============================================================================
# 1. PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Smart Academic Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 2. GLOBAL STYLE
# ============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    :root {
        --navy-900: #0B1220;
        --blue-900: #0F2557;
        --blue-800: #123A85;
        --blue-700: #1D4ED8;
        --blue-600: #2563EB;
        --blue-500: #3B82F6;
        --blue-300: #93C5FD;
        --blue-100: #DBEAFE;
        --bg: #F3F6FC;
        --surface: #FFFFFF;
        --border: #E2E8F5;
        --text-main: #0F172A;
        --text-soft: #64748B;
        --success: #16A34A;
        --danger: #DC2626;
        --warning: #D97706;
    }

    .stApp {
        background: var(--bg);
    }

    h1, h2, h3, h4, p, span, label, div {
        color: var(--text-main);
    }

    /* ===== Hero header ===== */
    .hero {
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        gap: 22px;
        padding: 34px 38px;
        border-radius: 22px;
        background: linear-gradient(120deg, var(--navy-900) 0%, var(--blue-900) 45%, var(--blue-600) 100%);
        box-shadow: 0 16px 40px rgba(15, 37, 87, 0.28);
        margin-bottom: 26px;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(147, 197, 253, 0.22) 0%, rgba(147, 197, 253, 0) 70%);
        border-radius: 50%;
    }
    .hero .icon-badge {
        font-size: 40px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 18px;
        padding: 14px 20px;
        z-index: 1;
    }
    .hero h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 26px;
        font-weight: 800;
    }
    .hero p {
        color: rgba(255,255,255,0.85) !important;
        margin: 6px 0 0 0;
        font-size: 14px;
        line-height: 1.5;
    }
    .hero .hero-badges { margin-top: 12px; z-index: 1; }
    .hero-chip {
        display: inline-block;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.25);
        color: #FFFFFF;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }

    /* ===== Page title (per-page) ===== */
    .page-title {
        font-size: 20px;
        font-weight: 800;
        color: var(--blue-900);
        margin-bottom: 2px;
    }
    .page-subtitle {
        font-size: 13.5px;
        color: var(--text-soft);
        margin-bottom: 20px;
    }

    /* ===== Fully-gradient metric cards ===== */
    .metric-card {
        border-radius: 18px;
        padding: 20px 22px;
        color: #FFFFFF !important;
        box-shadow: 0 10px 24px rgba(15, 37, 87, 0.18);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 34px rgba(15, 37, 87, 0.26);
    }
    .metric-card * { color: #FFFFFF !important; }
    .metric-card .metric-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px; height: 38px;
        border-radius: 12px;
        background: rgba(255,255,255,0.18);
        font-size: 18px;
        margin-bottom: 10px;
    }
    .metric-card .metric-value {
        font-size: 28px;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-card .metric-label {
        font-size: 12.5px;
        opacity: 0.9;
        margin-top: 4px;
    }
    .metric-card .metric-sub {
        font-size: 11px;
        opacity: 0.75;
        margin-top: 8px;
    }

    .grad-deep   { background: linear-gradient(135deg, var(--navy-900) 0%, var(--blue-800) 100%); }
    .grad-mid    { background: linear-gradient(135deg, var(--blue-900) 0%, var(--blue-600) 100%); }
    .grad-bright { background: linear-gradient(135deg, var(--blue-700) 0%, var(--blue-300) 100%); }
    .grad-soft   { background: linear-gradient(135deg, var(--blue-600) 0%, var(--blue-300) 100%); }

    /* ===== Panels / sections ===== */
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 6px 18px rgba(15, 37, 87, 0.05);
    }

    /* ===== Source / detail cards ===== */
    .source-card {
        background: var(--blue-100);
        border-left: 3px solid var(--blue-600);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        color: var(--blue-900) !important;
    }
    .source-card * { color: var(--blue-900) !important; }
    .source-card .source-title { font-weight: 700; margin-bottom: 4px; }

    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: linear-gradient(120deg, var(--blue-100), #EFF6FF);
        border: 1px solid var(--blue-300);
        color: var(--blue-900);
        border-radius: 10px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }
    .status-ok { background: rgba(22, 163, 74, 0.12); color: var(--success); }
    .status-bad { background: rgba(220, 38, 38, 0.12); color: var(--danger); }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: var(--blue-100);
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        color: var(--blue-900);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--blue-900), var(--blue-600)) !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero(title, subtitle, chips=None):
    chips_html = ""
    if chips:
        chips_html = "<div class='hero-badges'>" + "".join(
            f"<span class='hero-chip'>{c}</span>" for c in chips
        ) + "</div>"

    st.markdown(
        f"""
        <div class="hero">
            <div class="icon-badge">📚</div>
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
                {chips_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_title(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(column, value, label, icon, gradient="grad-mid", sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    column.markdown(
        f"""
        <div class="metric-card {gradient}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# 3. MODEL / CLASSIFIER ARCHITECTURE
# ============================================================================
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

    embedding_client = vector_stage.get_embedding_model()

    hf_token = st.secrets.get("HF_TOKEN")
    if not hf_token:
        st.error(
            "HF_TOKEN is missing from the app's Secrets.\n\n"
            "Go to Streamlit Cloud → App settings → Secrets and add a line like:\n\n"
            'HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"'
        )
        st.stop()

    llm_node = prompting_stage.build_llm(hf_token)

    return dl_model, token_generator, embedding_client, llm_node


if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    dl_model, tokenizer, embeddings, llm = initialize_system_resources()
else:
    st.error(
        f"'{MODEL_PATH}' or '{TOKENIZER_PATH}' is missing from the repo. "
        "Run train_model.py locally first, then upload both files (h5 + json) to GitHub "
        "next to app.py."
    )
    st.stop()


def classify_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )
    prediction = dl_model.predict(padded_sequence, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    return {
        "category": TOP_CATEGORIES[predicted_index],
        "confidence": float(prediction[predicted_index]) * 100,
        "distribution": {
            TOP_CATEGORIES[i]: float(prediction[i]) * 100 for i in range(len(TOP_CATEGORIES))
        },
    }


# ============================================================================
# 4. INGESTION PIPELINE (stages 1-2-3-5)
# ============================================================================
def execute_vector_ingestion():
    with st.spinner("Running document ETL: loading, splitting, and vector persistence..."):
        st.write("Step 1: Loading PDFs...")
        documents = documents_stage.load_documents(PAPERS_DIR)
        st.write(f"Loaded {len(documents)} document page(s).")

        if not documents:
            st.sidebar.error(f"'{PAPERS_DIR}' contains zero documents.")
            return None

        st.write("Step 2: Cleaning and splitting documents...")
        documents = preprocessing_stage.clean_documents(documents)
        chunks = chunking_stage.chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        st.write(f"Created {len(chunks)} chunk(s).")

        st.write("Step 3: Creating embeddings and storing in ChromaDB...")
        vector_store = chroma_stage.create_chroma_store(
            chunks=chunks,
            embedding_model=embeddings,
            persist_directory=DB_DIR,
        )
        st.write("Step 4: Done.")

        st.sidebar.success(f"Ingestion complete — {len(chunks)} chunk(s) persisted.")
        return vector_store


def get_indexed_pdf_files():
    if not os.path.exists(PAPERS_DIR):
        return []
    return sorted(f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf"))


# Vector DB connection (if it already exists on disk)
vector_db = None
if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    vector_db = chroma_stage.load_chroma_store(embeddings, DB_DIR)


def get_chunk_metadatas():
    if vector_db is None:
        return []
    try:
        raw = vector_db._collection.get(include=["metadatas"])
        return [m for m in raw["metadatas"] if m]
    except Exception:
        return []


def get_total_chunks():
    if vector_db is None:
        return 0
    try:
        return vector_db._collection.count()
    except Exception:
        return 0


# ============================================================================
# 5. SESSION STATE
# ============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "classification_log" not in st.session_state:
    st.session_state.classification_log = []


# ============================================================================
# 6. SIDEBAR — NAVIGATION + FILE UPLOAD + CONTROL PANEL
# ============================================================================
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📁 Documents", "📊 Analytics", "🎯 Answer Accuracy", "🧠 Classification"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📤 Upload New Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        new_files_saved = 0
        for uploaded_file in uploaded_files:
            destination_path = os.path.join(PAPERS_DIR, uploaded_file.name)
            if not os.path.exists(destination_path):
                with open(destination_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_saved += 1

        if new_files_saved:
            st.success(f"✅ Saved {new_files_saved} new file(s) to '{PAPERS_DIR}/'.")

    current_files = get_indexed_pdf_files()
    if current_files:
        st.caption(f"📄 {len(current_files)} file(s) ready for ingestion:")
        chips_html = "".join(f'<span class="file-chip">📄 {f}</span>' for f in current_files)
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.caption("No files in the folder yet.")

    st.markdown("---")
    st.markdown("### ⚙️ System Control Panel")
    st.markdown(
        f"**Instructions:**\n"
        f"1. Upload PDF files above (or place them manually in `{PAPERS_DIR}/`).\n"
        "2. Run the ingestion pipeline below."
    )

    if st.button("🔄 Execute Ingestion Pipeline", use_container_width=True):
        execute_vector_ingestion()
        st.rerun()

    st.markdown("---")
    db_status_html = (
        '<span class="status-pill status-ok">🟢 Vector DB Ready</span>'
        if vector_db is not None
        else '<span class="status-pill status-bad">🔴 Not Indexed Yet</span>'
    )
    st.markdown(db_status_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Infrastructure Specifications")
    st.info(
        "• DL Intent Engine: **LSTM (Keras Backend)**\n"
        "• Vector DB: **ChromaDB**\n"
        "• Embedding Model: **all-MiniLM-L6-v2 (HuggingFace)**\n"
        "• Generative LLM: **Qwen2.5-7B-Instruct (HuggingFace Inference Providers)**"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================================
# 7. PAGE: HOME (main chat experience)
# ============================================================================
def page_home():
    render_hero(
        "Smart Academic Research Assistant",
        "Hybrid RAG + Deep Learning Framework — combining a domain classifier with "
        "retrieval-augmented generation over your own research papers.",
        chips=["LSTM Classifier", "ChromaDB", "Qwen2.5-7B", "HuggingFace Embeddings"],
    )

    col1, col2, col3 = st.columns(3)
    metric_card(col1, len(get_indexed_pdf_files()), "Documents Uploaded", "📄", "grad-deep")
    metric_card(col2, get_total_chunks(), "Chunks Indexed", "🧩", "grad-mid")
    metric_card(
        col3,
        len(st.session_state.chat_history),
        "Questions This Session",
        "💬",
        "grad-bright",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    render_page_title("💬 Ask a Question", "Ask a research question or paste an abstract below.")

    for entry in st.session_state.chat_history:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(entry["query"])

        with st.chat_message("assistant", avatar="📚"):
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    label="Predicted Domain",
                    value=CATEGORY_LABELS.get(entry["predicted_class"], entry["predicted_class"]),
                )
            with metric_col2:
                st.metric(label="Classifier Confidence", value=f"{entry['confidence']:.2f}%")

            if entry.get("answer"):
                st.write(entry["answer"])
                if entry.get("sources"):
                    with st.expander(f"📄 Sources used ({len(entry['sources'])})"):
                        for src in entry["sources"]:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div class="source-title">📄 {src['file']} — page {src['page']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.info(src["content"])
            elif entry.get("warning"):
                st.warning(entry["warning"])
            elif entry.get("error"):
                st.error(entry["error"])

    user_query = st.chat_input("Ask a research question or paste an abstract here...")

    if user_query:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)

        classification = classify_text(user_query)
        predicted_class = classification["category"]
        confidence_score = classification["confidence"]

        st.session_state.classification_log.append({
            "timestamp": datetime.now(),
            "text": user_query,
            "category": predicted_class,
            "confidence": confidence_score,
            "source": "chat",
        })

        entry = {
            "query": user_query,
            "predicted_class": predicted_class,
            "confidence": confidence_score,
        }

        with st.chat_message("assistant", avatar="📚"):
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    label="Predicted Domain",
                    value=CATEGORY_LABELS.get(predicted_class, predicted_class),
                )
            with metric_col2:
                st.metric(label="Classifier Confidence", value=f"{confidence_score:.2f}%")

            if vector_db:
                retriever_node = retrieval_stage.get_retriever(vector_db)
                rag_orchestration_chain = prompting_stage.build_rag_chain(llm, retriever_node)

                with st.spinner("Retrieving relevant context and generating a response..."):
                    try:
                        start_time = time.time()
                        execution_response = prompting_stage.generate_answer(rag_orchestration_chain, user_query)
                        elapsed = time.time() - start_time

                        answer_text = execution_response["answer"]
                        st.write(answer_text)
                        st.caption(f"⏱️ {elapsed:.2f}s")

                        sources = []
                        for document in execution_response["context"]:
                            sources.append({
                                "file": os.path.basename(document.metadata.get("source", "Unknown_Reference.pdf")),
                                "page": document.metadata.get("page", "N/A"),
                                "content": document.page_content,
                            })

                        if sources:
                            with st.expander(f"📄 Sources used ({len(sources)})"):
                                for src in sources:
                                    st.markdown(
                                        f"""
                                        <div class="source-card">
                                            <div class="source-title">📄 {src['file']} — page {src['page']}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    st.info(src["content"])

                        entry["answer"] = answer_text
                        entry["sources"] = sources
                        entry["elapsed"] = elapsed
                        entry["not_found"] = NOT_AVAILABLE_PHRASE in answer_text.lower()
                    except Exception as e:
                        error_message = (
                            f"An error occurred while generating the answer: {e}\n\n"
                            "If the error mentions the model is unavailable (not supported / 404), "
                            "change the `LLM_REPO_ID` value in config.py to a model currently available at "
                            "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation"
                        )
                        st.error(error_message)
                        entry["error"] = error_message
            else:
                warning_message = (
                    "RAG pipeline is offline. Upload PDF files from the sidebar and run "
                    "the Ingestion Pipeline first."
                )
                st.warning(warning_message)
                entry["warning"] = warning_message

        st.session_state.chat_history.append(entry)


# ============================================================================
# 8. PAGE: DOCUMENTS
# ============================================================================
def page_documents():
    render_page_title("📁 Documents", "Manage the source PDFs that power your knowledge base.")

    col1, col2, col3 = st.columns(3)
    metric_card(col1, len(get_indexed_pdf_files()), "Total PDF Files", "📄", "grad-deep")
    metric_card(col2, get_total_chunks(), "Total Chunks", "🧩", "grad-mid")

    metadatas = get_chunk_metadatas()
    unique_files_indexed = len({m.get("source") for m in metadatas if m.get("source")})
    metric_card(col3, unique_files_indexed, "Files Actually Indexed", "🗂️", "grad-bright")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 📄 Document List")

        files = get_indexed_pdf_files()
        if not files:
            st.info("No documents uploaded yet. Use the uploader in the sidebar to add PDF files.")
        else:
            rows = []
            for f in files:
                file_path = os.path.join(PAPERS_DIR, f)
                size_kb = round(os.path.getsize(file_path) / 1024, 1)
                chunk_count = sum(
                    1 for m in metadatas
                    if os.path.basename(m.get("source", "")) == f
                )
                rows.append({
                    "File Name": f,
                    "Size (KB)": size_kb,
                    "Chunks Indexed": chunk_count,
                    "Status": "✅ Indexed" if chunk_count > 0 else "⏳ Pending ingestion",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 📈 Chunks per Document")
        if metadatas:
            filenames_list = [os.path.basename(m.get("source", "Unknown")) for m in metadatas]
            counts = pd.Series(filenames_list).value_counts().reset_index()
            counts.columns = ["File", "Chunks"]
            st.bar_chart(counts.set_index("File"))
        else:
            st.caption("No chunk data yet — run the ingestion pipeline to see this chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Ingestion Pipeline Steps")
    st.markdown(
        """
        1. **Load** — Read every PDF in `papers_to_chat/` using `PyPDFDirectoryLoader`.
        2. **Clean** — Normalize whitespace and drop blank pages (`02_preprocessing.py`).
        3. **Chunk** — Split into overlapping chunks of **{chunk_size}** characters
           (overlap **{overlap}**) using `RecursiveCharacterTextSplitter`.
        4. **Embed & Store** — Convert chunks to vectors with **all-MiniLM-L6-v2** and
           persist them in **ChromaDB** at `{db_dir}/`.
        """.format(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, db_dir=DB_DIR)
    )
    st.caption("Trigger this pipeline any time from the sidebar's 'Execute Ingestion Pipeline' button.")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# 9. PAGE: ANALYTICS
# ============================================================================
def page_analytics():
    render_page_title("📊 Analytics", "A high-level view of your knowledge base and usage patterns.")

    col1, col2, col3, col4 = st.columns(4)
    total_files = len(get_indexed_pdf_files())
    total_chunks = get_total_chunks()
    total_questions = len(st.session_state.chat_history)
    avg_chunks_per_file = round(total_chunks / total_files, 1) if total_files else 0

    metric_card(col1, total_files, "Documents", "📄", "grad-deep")
    metric_card(col2, total_chunks, "Chunks", "🧩", "grad-mid")
    metric_card(col3, total_questions, "Questions Asked", "💬", "grad-bright")
    metric_card(col4, avg_chunks_per_file, "Avg. Chunks / File", "📐", "grad-soft")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 🗂️ Document Distribution")
        metadatas = get_chunk_metadatas()
        if metadatas:
            filenames_list = [os.path.basename(m.get("source", "Unknown")) for m in metadatas]
            counts = pd.Series(filenames_list).value_counts().reset_index()
            counts.columns = ["File", "Chunks"]
            st.bar_chart(counts.set_index("File"))
        else:
            st.caption("No indexed data yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Classified Topics This Session")
        if st.session_state.classification_log:
            cats = [
                CATEGORY_LABELS.get(c["category"], c["category"])
                for c in st.session_state.classification_log
            ]
            counts = pd.Series(cats).value_counts().reset_index()
            counts.columns = ["Domain", "Count"]
            st.bar_chart(counts.set_index("Domain"))
        else:
            st.caption("Ask a question to see topic classification analytics here.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🕐 Session Activity Log")
    if st.session_state.classification_log:
        log_rows = [
            {
                "Time": c["timestamp"].strftime("%H:%M:%S"),
                "Query": c["text"][:80] + ("..." if len(c["text"]) > 80 else ""),
                "Predicted Domain": CATEGORY_LABELS.get(c["category"], c["category"]),
                "Confidence": f"{c['confidence']:.1f}%",
            }
            for c in reversed(st.session_state.classification_log)
        ]
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No activity yet this session.")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# 10. PAGE: ANSWER ACCURACY
# ============================================================================
def page_answer_accuracy():
    render_page_title("🎯 Answer Accuracy", "Quality signals for the answers generated by the RAG pipeline.")

    answered_entries = [e for e in st.session_state.chat_history if e.get("answer")]
    total_answered = len(answered_entries)
    not_found_count = sum(1 for e in answered_entries if e.get("not_found"))
    avg_elapsed = (
        round(sum(e.get("elapsed", 0) for e in answered_entries) / total_answered, 2)
        if total_answered else 0
    )
    avg_sources = (
        round(sum(len(e.get("sources", [])) for e in answered_entries) / total_answered, 1)
        if total_answered else 0
    )
    grounded_rate = (
        round((total_answered - not_found_count) / total_answered * 100, 1)
        if total_answered else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    metric_card(col1, total_answered, "Answers Generated", "✍️", "grad-deep")
    metric_card(col2, f"{grounded_rate}%", "Grounded in Context", "✅", "grad-mid", sub="Answer was not 'not available'")
    metric_card(col3, f"{avg_elapsed}s", "Avg. Response Time", "⏱️", "grad-bright")
    metric_card(col4, avg_sources, "Avg. Sources per Answer", "📎", "grad-soft")

    st.markdown("<br>", unsafe_allow_html=True)

    if not answered_entries:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.info(
            "No generated answers yet. Ask a few questions on the Home page — this page will "
            "then show response time, grounding rate, and source usage across your session."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### ⏱️ Response Time per Question")
        timing_df = pd.DataFrame({
            "Question #": list(range(1, total_answered + 1)),
            "Seconds": [round(e.get("elapsed", 0), 2) for e in answered_entries],
        })
        st.line_chart(timing_df.set_index("Question #"))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 📎 Sources Retrieved per Question")
        sources_df = pd.DataFrame({
            "Question #": list(range(1, total_answered + 1)),
            "Sources": [len(e.get("sources", [])) for e in answered_entries],
        })
        st.bar_chart(sources_df.set_index("Question #"))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Answer Grounding Detail")
    st.caption(
        "An answer is flagged 'Not grounded' when the model explicitly said the requested "
        "information isn't available in the ingested references — this means the retrieved "
        "context didn't contain a confident answer, rather than the model guessing."
    )
    detail_rows = [
        {
            "Query": e["query"][:70] + ("..." if len(e["query"]) > 70 else ""),
            "Response Time": f"{e.get('elapsed', 0):.2f}s",
            "Sources Used": len(e.get("sources", [])),
            "Grounded": "❌ Not grounded" if e.get("not_found") else "✅ Grounded",
        }
        for e in answered_entries
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# 11. PAGE: CLASSIFICATION
# ============================================================================
def page_classification():
    render_page_title(
        "🧠 Classification",
        "Test the standalone LSTM domain classifier — independent from the RAG answer generation.",
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🔤 Try the Classifier")
    st.caption(
        "Paste any research abstract or question. The LSTM model predicts which of the 4 "
        "trained academic domains it belongs to, with a confidence score for each."
    )

    sample_text = st.text_area(
        "Text to classify",
        placeholder="Paste an abstract or research question here...",
        height=120,
    )

    if st.button("🔎 Classify Text", type="primary"):
        if sample_text.strip():
            result = classify_text(sample_text)
            st.session_state.classification_log.append({
                "timestamp": datetime.now(),
                "text": sample_text,
                "category": result["category"],
                "confidence": result["confidence"],
                "source": "classification_page",
            })

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Predicted Domain",
                    CATEGORY_LABELS.get(result["category"], result["category"]),
                )
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2f}%")

            st.markdown("##### Probability Distribution")
            dist_df = pd.DataFrame({
                "Domain": [CATEGORY_LABELS[c] for c in TOP_CATEGORIES],
                "Probability (%)": [result["distribution"][c] for c in TOP_CATEGORIES],
            })
            st.bar_chart(dist_df.set_index("Domain"))
        else:
            st.warning("Please enter some text to classify.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 📚 Trained Domains")
    domain_cols = st.columns(4)
    domain_descriptions = {
        "cs": "Computer science, machine learning, algorithms, and software systems.",
        "math": "Pure and applied mathematics, proofs, and mathematical modeling.",
        "physics": "General physics research, excluding astrophysics-specific topics.",
        "astro-ph": "Astrophysics, cosmology, and observational astronomy.",
    }
    gradients = ["grad-deep", "grad-mid", "grad-bright", "grad-soft"]
    for i, cat in enumerate(TOP_CATEGORIES):
        metric_card(
            domain_cols[i],
            CATEGORY_LABELS[cat],
            domain_descriptions[cat],
            "🔖",
            gradients[i % len(gradients)],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🏗️ Model Architecture")
    st.code(
        f"""Input (max_length={MAX_SEQUENCE_LENGTH})
→ Embedding(vocab_size={VOCAB_SIZE}, dim={EMBEDDING_DIM})
→ LSTM(64, return_sequences=True)
→ Dropout(0.3)
→ LSTM(32)
→ Dense(32, activation='relu')
→ Dropout(0.2)
→ Dense({len(TOP_CATEGORIES)}, activation='softmax')""",
        language="text",
    )
    st.caption("Trained in train_model.py and loaded here via load_weights() for cross-version compatibility.")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.classification_log:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 📜 Classification History (this session)")
        log_rows = [
            {
                "Time": c["timestamp"].strftime("%H:%M:%S"),
                "Text": c["text"][:70] + ("..." if len(c["text"]) > 70 else ""),
                "Domain": CATEGORY_LABELS.get(c["category"], c["category"]),
                "Confidence": f"{c['confidence']:.1f}%",
                "Source": "Chat" if c["source"] == "chat" else "Classification Page",
            }
            for c in reversed(st.session_state.classification_log)
        ]
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# 12. ROUTER
# ============================================================================
if page == "🏠 Home":
    page_home()
elif page == "📁 Documents":
    page_documents()
elif page == "📊 Analytics":
    page_analytics()
elif page == "🎯 Answer Accuracy":
    page_answer_accuracy()
elif page == "🧠 Classification":
    page_classification()
