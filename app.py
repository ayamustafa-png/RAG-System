import sys

# --- 0. SQLite3 Workaround (required on Streamlit Community Cloud for ChromaDB) ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import time
import math
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
CATEGORY_COLORS = {
    "cs": "#818CF8",
    "math": "#4338CA",
    "physics": "#2DD4BF",
    "astro-ph": "#0D9488",
}
NOT_AVAILABLE_PHRASE = "not available"

PAGES = [
    ("🛰️", "Mission Control"),
    ("🗃️", "Archive"),
    ("📡", "Signal Analytics"),
    ("🎯", "Precision Metrics"),
    ("🧬", "Domain Scanner"),
]

# ============================================================================
# 1. PAGE CONFIG
# ============================================================================
def _load_page_icon():
    """بيحاول يفتح لوجو Scholar AI كصورة لاستخدامه كـ page icon، ولو مش موجود
    (لسه ما اترفعش للريبو) بيرجع إيموجي بديل عشان التطبيق يفضل شغال من غير كسر."""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scholar_ai_icon.png")
    if os.path.exists(logo_path):
        from PIL import Image
        return Image.open(logo_path)
    return "📚"


st.set_page_config(
    page_title="Scholar AI",
    page_icon=_load_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Define your background image URL
background_image_url = "https://en.wikipedia.org/wiki/Typing#/media/File:Computer_keyboard.png"

# 2. Inject CSS with st.markdown
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================================
# 2. GLOBAL STYLE — "Mission Control" theme
# ============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    :root {
        --cyan: #6366F1;
        --violet: #4338CA;
        --amber: #2DD4BF;
        --pink: #0D9488;
        --brand-navy: #1E1B4B;
        --brand-indigo: #4338CA;
        --brand-teal: #0D9488;
        --brand-teal-light: #2DD4BF;
        --brand-surface: #F8FAFC;
        --ink: #1E1B4B;
        --panel: #FFFFFF;
        --panel-border: #E4E1F7;
        --text-main: #1E1B4B;
        --text-soft: #666B8C;
        --success: #0D9488;
        --danger: #E11D48;
    }

    .stApp {
        background: linear-gradient(160deg, #F5F3FF 0%, #F1F5FF 45%, #ECFDF9 100%);
        background-attachment: fixed;
    }

    h1, h2, h3, h4, p, span, label, div {
        color: var(--text-main);
    }

    section[data-testid="stSidebar"] {
        background: #FDFCFF;
        border-right: 1px solid var(--panel-border);
    }
    section[data-testid="stSidebar"] * { color: var(--text-main) !important; }

    /* ===== HUD hero strip ===== */
    .hud {
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        padding: 28px 34px;
        margin-bottom: 20px;
        background: linear-gradient(115deg, var(--brand-navy) 0%, var(--brand-indigo) 55%, var(--brand-teal) 100%);
        box-shadow: 0 16px 36px rgba(67, 56, 202, 0.28);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hud::before {
        content: "";
        position: absolute; inset: 0;
        background: repeating-linear-gradient(90deg, transparent, transparent 38px, rgba(255,255,255,0.04) 39px, transparent 40px);
        pointer-events: none;
    }
    .hud-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    /* Solid, high-contrast colors instead of background-clip gradient text —
       gradient-clip text can render invisible under some browsers, print
       views, or forced dark-mode extensions. Solid colors always stay legible. */
    .hud-title .brand-scholar { color: #FFFFFF !important; }
    .hud-title .brand-ai { color: var(--brand-teal-light) !important; }
    .hud-logo {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        flex-shrink: 0;
    }
    .hud-sub {
        color: rgba(255,255,255,0.82) !important;
        font-size: 13px;
        margin-top: 8px;
        line-height: 1.6;
        max-width: 720px;
    }
    .hud-tags { margin-top: 14px; }
    .hud-tag {
        display: inline-block;
        font-size: 10.5px;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.22);
        background: rgba(255,255,255,0.10);
        color: #FFFFFF !important;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px 6px 3px 0;
    }

    /* ===== Section dividers (HUD style) ===== */
    .section-divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 26px 0 14px 0;
    }
    .section-divider span {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        letter-spacing: 2px;
        color: var(--brand-indigo) !important;
        white-space: nowrap;
    }
    .section-divider::after {
        content: "";
        flex: 1;
        height: 1px;
        background: repeating-linear-gradient(90deg, var(--panel-border), var(--panel-border) 4px, transparent 4px, transparent 8px);
    }

    /* ===== Glass panels ===== */
    .glass {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 8px 24px rgba(67, 56, 202, 0.07);
    }

    /* ===== Stat pods — full two-tone gradients, not just an edge line ===== */
    .pod {
        position: relative;
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid rgba(255,255,255,0.12);
        height: 100%;
        overflow: hidden;
        box-shadow: 0 10px 26px rgba(30, 27, 75, 0.16);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .pod:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 34px rgba(30, 27, 75, 0.24);
    }
    .pod-cyan   { background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%); }
    .pod-violet { background: linear-gradient(135deg, #1E1B4B 0%, #4338CA 100%); }
    .pod-amber  { background: linear-gradient(135deg, #2DD4BF 0%, #0D9488 100%); }
    .pod-pink   { background: linear-gradient(135deg, #0D9488 0%, #1E1B4B 100%); }
    .pod-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .pod-label {
        font-size: 11px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.82) !important;
        margin-top: 4px;
    }
    .pod-sub {
        font-size: 10.5px;
        color: rgba(255,255,255,0.68) !important;
        margin-top: 6px;
    }

    /* ===== Signal bars (custom horizontal chart) ===== */
    .signal-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .signal-label {
        width: 150px;
        font-size: 11.5px;
        color: var(--text-soft) !important;
        flex-shrink: 0;
    }
    .signal-track {
        flex: 1;
        height: 10px;
        border-radius: 6px;
        background: #EEECFB;
        overflow: hidden;
    }
    .signal-fill {
        height: 100%;
        border-radius: 6px;
    }
    .signal-value {
        width: 52px;
        text-align: right;
        font-size: 11.5px;
        color: var(--text-main) !important;
        flex-shrink: 0;
    }

    /* ===== Terminal log rows ===== */
    .term-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 8px;
        font-size: 12px;
        margin-bottom: 4px;
        background: #FFFFFF;
        border: 1px solid var(--panel-border);
    }
    .term-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-top: 4px;
        flex-shrink: 0;
    }
    .dot-ok { background: var(--success); box-shadow: 0 0 6px var(--success); }
    .dot-bad { background: var(--danger); box-shadow: 0 0 6px var(--danger); }
    .dot-neutral { background: var(--brand-indigo); box-shadow: 0 0 6px var(--brand-indigo); }
    .term-time {
        color: var(--text-soft) !important;
        width: 68px;
        flex-shrink: 0;
    }
    .term-text { color: var(--text-main) !important; flex: 1; }
    .term-badge {
        font-size: 10px;
        padding: 2px 8px;
        border-radius: 5px;
        background: #EEECFB;
        color: var(--brand-indigo) !important;
        flex-shrink: 0;
    }

    /* ===== File cartridge chips ===== */
    .cartridge {
        display: inline-flex;
        flex-direction: column;
        gap: 2px;
        border: 1px solid var(--panel-border);
        background: linear-gradient(160deg, #F0F0FE, #FFFFFF);
        border-radius: 12px;
        padding: 10px 14px;
        margin: 4px;
        min-width: 190px;
    }
    .cartridge .c-name { font-size: 11.5px; color: var(--text-main) !important; }
    .cartridge .c-meta { font-size: 10px; color: var(--text-soft) !important; }

    /* ===== Status readout ===== */
    .readout {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 11.5px;
        letter-spacing: 0.4px;
        padding: 6px 14px;
        border-radius: 999px;
        border: 1px solid var(--panel-border);
        background: #FFFFFF;
    }
    .readout .rdot { width: 8px; height: 8px; border-radius: 50%; }

    /* ===== Nav buttons ===== */
    div[data-testid="stButton"] button {
        border-radius: 10px !important;
        border: 1px solid var(--panel-border) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12.5px !important;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SMALL RENDER HELPERS (custom SVG / HTML — no default chart widgets)
# ============================================================================
def get_logo_data_uri():
    """بتقرا لوجو Scholar AI من الديسك وتحوّله base64 لعرضه جوا الـ HTML مباشرة.
    لو الملف مش موجود لسه (متضافش للريبو)، بترجع None ويتم استخدام إيموجي بديل."""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scholar_ai_icon.png")
    if not os.path.exists(logo_path):
        return None
    import base64
    with open(logo_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_hud(tags=None):
    tags_html = ""
    if tags:
        tags_html = "<div class='hud-tags'>" + "".join(f"<span class='hud-tag'>{t}</span>" for t in tags) + "</div>"

    logo_uri = get_logo_data_uri()
    logo_html = (
        f'<img src="{logo_uri}" class="hud-logo" alt="Scholar AI logo"/>'
        if logo_uri else "📚"
    )

    st.markdown(
        f"""
        <div class="hud">
            <div class="hud-title">{logo_html}<span class="brand-scholar">Scholar</span><span class="brand-ai">AI</span></div>
            <div class="hud-sub">Hybrid deep-learning classification + retrieval-augmented generation,
            running live against your own indexed research papers.</div>
            {tags_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title):
    st.markdown(f'<div class="section-divider"><span>⟨ {title.upper()} ⟩</span></div>', unsafe_allow_html=True)


def pod(column, value, label, accent="cyan", sub=None):
    sub_html = f'<div class="pod-sub">{sub}</div>' if sub else ""
    column.markdown(
        f"""
        <div class="pod pod-{accent}">
            <div class="pod-value">{value}</div>
            <div class="pod-label">{label}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def svg_gauge(percentage, label="", color="#818CF8", size=150):
    percentage = max(0, min(100, percentage))
    radius = 52
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - percentage / 100)
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r="{radius}" stroke="rgba(255,255,255,0.08)" stroke-width="9" fill="none"/>
      <circle cx="60" cy="60" r="{radius}" stroke="{color}" stroke-width="9" fill="none"
        stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" stroke-linecap="round"
        transform="rotate(-90 60 60)" style="filter: drop-shadow(0 0 6px {color});"/>
      <text x="60" y="56" text-anchor="middle" font-family="Space Grotesk, sans-serif"
        font-size="22" font-weight="700" fill="#FFFFFF">{percentage:.0f}%</text>
      <text x="60" y="76" text-anchor="middle" font-family="JetBrains Mono, monospace"
        font-size="9" fill="rgba(231,236,251,0.65)">{label}</text>
    </svg>
    """


def signal_bars(items):
    """items: list of (label, value 0-100, color)"""
    rows = ""
    for label, value, color in items:
        width = max(2, min(100, value))
        rows += f"""
        <div class="signal-row">
            <div class="signal-label">{label}</div>
            <div class="signal-track">
                <div class="signal-fill" style="width:{width}%; background: linear-gradient(90deg, {color}55, {color});"></div>
            </div>
            <div class="signal-value">{value:.1f}%</div>
        </div>
        """
    st.markdown(rows, unsafe_allow_html=True)


def sparkline(values, color="#818CF8", width=560, height=90):
    if not values:
        return ""
    n = len(values)
    vmax = max(values) if max(values) > 0 else 1
    vmin = min(values)
    span = (vmax - vmin) or 1
    step = width / max(1, n - 1)
    points = []
    for i, v in enumerate(values):
        x = i * step
        y = height - ((v - vmin) / span) * (height - 20) - 10
        points.append(f"{x:.1f},{y:.1f}")
    path = " ".join(points)
    dots = "".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="3" fill="{color}"/>' for p in points
    )
    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"
          style="filter: drop-shadow(0 0 4px {color});"/>
        {dots}
    </svg>
    """


def render_terminal_log(rows):
    """rows: list of dicts with keys: dot ('ok'|'bad'|'neutral'), time, text, badge"""
    html = ""
    for r in rows:
        html += f"""
        <div class="term-row">
            <div class="term-dot dot-{r['dot']}"></div>
            <div class="term-time">{r['time']}</div>
            <div class="term-text">{r['text']}</div>
            <div class="term-badge">{r['badge']}</div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


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


@st.cache_resource(show_spinner="Booting mission systems...")
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

if "active_page" not in st.session_state:
    st.session_state.active_page = "Mission Control"


# ============================================================================
# 6. SIDEBAR — CONTROL DECK
# ============================================================================
with st.sidebar:
    _sidebar_logo_uri = get_logo_data_uri()
    _sidebar_logo_html = (
        f'<img src="{_sidebar_logo_uri}" style="width:28px;height:28px;border-radius:7px;vertical-align:middle;margin-right:8px;"/>'
        if _sidebar_logo_uri else "🛠️ "
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0;font-size:19px;font-weight:700;'
        f'font-family:\'Space Grotesk\',sans-serif;margin-bottom:2px;">{_sidebar_logo_html}Control Deck</div>',
        unsafe_allow_html=True,
    )
    st.caption("Document intake & indexing operations.")

    uploaded_files = st.file_uploader(
        "Upload PDF file(s)",
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
            st.success(f"✅ {new_files_saved} file(s) saved to '{PAPERS_DIR}/'.")

    current_files = get_indexed_pdf_files()
    st.caption(f"📄 {len(current_files)} file(s) staged")

    if st.button("🚀 Execute Ingestion Pipeline", use_container_width=True):
        execute_vector_ingestion()
        st.rerun()

    st.markdown("---")

    ok = vector_db is not None
    dot_color = "#34D399" if ok else "#FB7185"
    status_text = "VECTOR DB ONLINE" if ok else "VECTOR DB OFFLINE"
    st.markdown(
        f"""<div class="readout"><span class="rdot" style="background:{dot_color}; box-shadow:0 0 6px {dot_color};"></span>{status_text}</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**System Stack**")
    st.caption("Classifier · LSTM (Keras)")
    st.caption("Vector Store · ChromaDB")
    st.caption("Embeddings · all-MiniLM-L6-v2")
    st.caption("Generator · Qwen2.5-7B-Instruct")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================================
# 7. TOP NAVIGATION (segmented control instead of sidebar radio)
# ============================================================================
render_hud(tags=["LSTM Classifier", "ChromaDB", "Qwen2.5-7B", "HuggingFace Embeddings", "RAG"])

nav_cols = st.columns(len(PAGES))
for i, (icon, name) in enumerate(PAGES):
    is_active = st.session_state.active_page == name
    if nav_cols[i].button(
        f"{icon} {name}",
        key=f"nav_{name}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.active_page = name
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# 8. PAGE: MISSION CONTROL (main chat experience)
# ============================================================================
def page_mission_control():
    col1, col2, col3 = st.columns(3)
    pod(col1, len(get_indexed_pdf_files()), "Documents Loaded", "cyan")
    pod(col2, get_total_chunks(), "Vectors Indexed", "violet")
    pod(col3, len(st.session_state.chat_history), "Transmissions This Session", "amber")

    render_section("Live Console")

    for entry in st.session_state.chat_history:
        with st.chat_message("user", avatar="🧑‍🚀"):
            st.markdown(entry["query"])

        with st.chat_message("assistant", avatar="🛰️"):
            gauge_col, info_col = st.columns([1, 3])
            with gauge_col:
                st.markdown(
                    svg_gauge(
                        entry["confidence"],
                        label=CATEGORY_LABELS.get(entry["predicted_class"], entry["predicted_class"]),
                        color=CATEGORY_COLORS.get(entry["predicted_class"], "#818CF8"),
                        size=110,
                    ),
                    unsafe_allow_html=True,
                )
            with info_col:
                if entry.get("answer"):
                    st.write(entry["answer"])
                    if entry.get("elapsed") is not None:
                        st.caption(f"⏱ {entry['elapsed']:.2f}s response time")
                    if entry.get("sources"):
                        with st.expander(f"📎 Sources ({len(entry['sources'])})"):
                            for src in entry["sources"]:
                                st.markdown(
                                    f"**{src['file']}** — page {src['page']}",
                                )
                                st.caption(src["content"][:400])
                elif entry.get("warning"):
                    st.warning(entry["warning"])
                elif entry.get("error"):
                    st.error(entry["error"])

    user_query = st.chat_input("Transmit a research question or paste an abstract...")

    if user_query:
        with st.chat_message("user", avatar="🧑‍🚀"):
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

        with st.chat_message("assistant", avatar="🛰️"):
            gauge_col, info_col = st.columns([1, 3])
            with gauge_col:
                st.markdown(
                    svg_gauge(
                        confidence_score,
                        label=CATEGORY_LABELS.get(predicted_class, predicted_class),
                        color=CATEGORY_COLORS.get(predicted_class, "#818CF8"),
                        size=110,
                    ),
                    unsafe_allow_html=True,
                )

            with info_col:
                if vector_db:
                    retriever_node = retrieval_stage.get_retriever(vector_db)
                    rag_orchestration_chain = prompting_stage.build_rag_chain(llm, retriever_node)

                    with st.spinner("Scanning archive and composing response..."):
                        try:
                            start_time = time.time()
                            execution_response = prompting_stage.generate_answer(rag_orchestration_chain, user_query)
                            elapsed = time.time() - start_time

                            answer_text = execution_response["answer"]
                            st.write(answer_text)
                            st.caption(f"⏱ {elapsed:.2f}s response time")

                            sources = []
                            for document in execution_response["context"]:
                                sources.append({
                                    "file": os.path.basename(document.metadata.get("source", "Unknown_Reference.pdf")),
                                    "page": document.metadata.get("page", "N/A"),
                                    "content": document.page_content,
                                })

                            if sources:
                                with st.expander(f"📎 Sources ({len(sources)})"):
                                    for src in sources:
                                        st.markdown(f"**{src['file']}** — page {src['page']}")
                                        st.caption(src["content"][:400])

                            entry["answer"] = answer_text
                            entry["sources"] = sources
                            entry["elapsed"] = elapsed
                            entry["not_found"] = NOT_AVAILABLE_PHRASE in answer_text.lower()
                        except Exception as e:
                            error_message = (
                                f"An error occurred while generating the answer: {e}\n\n"
                                "If the error mentions the model is unavailable (not supported / 404), "
                                "change `LLM_REPO_ID` in config.py to a model currently available at "
                                "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation"
                            )
                            st.error(error_message)
                            entry["error"] = error_message
                else:
                    warning_message = (
                        "RAG pipeline is offline. Upload PDF files from the Control Deck and run "
                        "the Ingestion Pipeline first."
                    )
                    st.warning(warning_message)
                    entry["warning"] = warning_message

        st.session_state.chat_history.append(entry)


# ============================================================================
# 9. PAGE: ARCHIVE (documents)
# ============================================================================
def page_archive():
    metadatas = get_chunk_metadatas()

    col1, col2, col3 = st.columns(3)
    pod(col1, len(get_indexed_pdf_files()), "Files in Archive", "cyan")
    pod(col2, get_total_chunks(), "Total Vectors", "violet")
    unique_files_indexed = len({m.get("source") for m in metadatas if m.get("source")})
    pod(col3, unique_files_indexed, "Files Actually Indexed", "amber")

    render_section("Document Cartridges")

    files = get_indexed_pdf_files()
    if not files:
        st.info("No documents loaded yet. Use the Control Deck uploader to add PDF files.")
    else:
        cartridges_html = ""
        for f in files:
            file_path = os.path.join(PAPERS_DIR, f)
            size_kb = round(os.path.getsize(file_path) / 1024, 1)
            chunk_count = sum(1 for m in metadatas if os.path.basename(m.get("source", "")) == f)
            status = "✅ indexed" if chunk_count > 0 else "⏳ pending"
            cartridges_html += f"""
            <div class="cartridge">
                <div class="c-name">📄 {f}</div>
                <div class="c-meta">{size_kb} KB · {chunk_count} chunks · {status}</div>
            </div>
            """
        st.markdown(cartridges_html, unsafe_allow_html=True)

    render_section("Chunk Distribution")

    if metadatas:
        filenames_list = [os.path.basename(m.get("source", "Unknown")) for m in metadatas]
        counts = pd.Series(filenames_list).value_counts()
        total = counts.sum()
        palette = ["#818CF8", "#4338CA", "#2DD4BF", "#0D9488", "#34D399"]
        items = [
            (name[:22] + ("…" if len(name) > 22 else ""), (count / total) * 100, palette[i % len(palette)])
            for i, (name, count) in enumerate(counts.items())
        ]
        signal_bars(items)
    else:
        st.caption("No chunk data yet — run the ingestion pipeline to populate this view.")

    render_section("Ingestion Pipeline")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(
        f"""
        **01 · Load** — Read every PDF in `{PAPERS_DIR}/` via `PyPDFDirectoryLoader`.

        **02 · Clean** — Normalize whitespace, drop blank pages.

        **03 · Chunk** — Split into overlapping segments of **{CHUNK_SIZE}** chars (overlap **{CHUNK_OVERLAP}**).

        **04 · Embed & Store** — Vectorize with **all-MiniLM-L6-v2**, persist to **ChromaDB** at `{DB_DIR}/`.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# 10. PAGE: SIGNAL ANALYTICS
# ============================================================================
def page_signal_analytics():
    total_files = len(get_indexed_pdf_files())
    total_chunks = get_total_chunks()
    total_questions = len(st.session_state.chat_history)
    avg_chunks_per_file = round(total_chunks / total_files, 1) if total_files else 0

    col1, col2, col3, col4 = st.columns(4)
    pod(col1, total_files, "Documents", "cyan")
    pod(col2, total_chunks, "Chunks", "violet")
    pod(col3, total_questions, "Questions Asked", "amber")
    pod(col4, avg_chunks_per_file, "Avg Chunks / File", "pink")

    render_section("Document Distribution")
    metadatas = get_chunk_metadatas()
    if metadatas:
        filenames_list = [os.path.basename(m.get("source", "Unknown")) for m in metadatas]
        counts = pd.Series(filenames_list).value_counts()
        total = counts.sum()
        palette = ["#818CF8", "#4338CA", "#2DD4BF", "#0D9488", "#34D399"]
        items = [
            (name[:22] + ("…" if len(name) > 22 else ""), (count / total) * 100, palette[i % len(palette)])
            for i, (name, count) in enumerate(counts.items())
        ]
        signal_bars(items)
    else:
        st.caption("No indexed data yet.")

    render_section("Classified Topics — This Session")
    if st.session_state.classification_log:
        cats = [c["category"] for c in st.session_state.classification_log]
        counts = pd.Series(cats).value_counts()
        total = counts.sum()
        items = [
            (CATEGORY_LABELS.get(cat, cat), (count / total) * 100, CATEGORY_COLORS.get(cat, "#818CF8"))
            for cat, count in counts.items()
        ]
        signal_bars(items)
    else:
        st.caption("Ask a question to populate topic analytics.")

    render_section("Session Activity Log")
    if st.session_state.classification_log:
        rows = [
            {
                "dot": "neutral",
                "time": c["timestamp"].strftime("%H:%M:%S"),
                "text": c["text"][:90] + ("..." if len(c["text"]) > 90 else ""),
                "badge": CATEGORY_LABELS.get(c["category"], c["category"]),
            }
            for c in reversed(st.session_state.classification_log)
        ]
        render_terminal_log(rows)
    else:
        st.caption("No activity yet this session.")


# ============================================================================
# 11. PAGE: PRECISION METRICS (answer accuracy)
# ============================================================================
def page_precision_metrics():
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

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="glass" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown(svg_gauge(grounded_rate, label="GROUNDED", color="#34D399", size=170), unsafe_allow_html=True)
        st.caption("Share of answers grounded in retrieved context (not flagged 'not available').")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        sub1, sub2, sub3 = st.columns(3)
        pod(sub1, total_answered, "Answers Generated", "cyan")
        pod(sub2, f"{avg_elapsed}s", "Avg Response Time", "violet")
        pod(sub3, avg_sources, "Avg Sources / Answer", "amber")

    if not answered_entries:
        render_section("No Data Yet")
        st.info(
            "No generated answers yet. Ask a few questions in Mission Control — this page "
            "will then chart response time, grounding rate, and source usage."
        )
        return

    render_section("Response Time Trend")
    timings = [round(e.get("elapsed", 0), 2) for e in answered_entries]
    st.markdown(sparkline(timings, color="#818CF8"), unsafe_allow_html=True)
    st.caption(f"Across {len(timings)} answered question(s), most recent on the right.")

    render_section("Answer Grounding Detail")
    rows = [
        {
            "dot": "bad" if e.get("not_found") else "ok",
            "time": f"{e.get('elapsed', 0):.2f}s",
            "text": e["query"][:80] + ("..." if len(e["query"]) > 80 else ""),
            "badge": f"{len(e.get('sources', []))} src" if not e.get("not_found") else "no match",
        }
        for e in answered_entries
    ]
    render_terminal_log(rows)


# ============================================================================
# 12. PAGE: DOMAIN SCANNER (classification)
# ============================================================================
def page_domain_scanner():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("**Scan any text against the standalone LSTM domain classifier** — independent from RAG generation.")

    sample_text = st.text_area(
        "Text to scan",
        placeholder="Paste an abstract or research question here...",
        height=120,
        label_visibility="collapsed",
    )

    if st.button("🧬 Run Scan", type="primary"):
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
            gauge_col, dist_col = st.columns([1, 2])
            with gauge_col:
                st.markdown(
                    svg_gauge(
                        result["confidence"],
                        label=CATEGORY_LABELS.get(result["category"], result["category"]),
                        color=CATEGORY_COLORS.get(result["category"], "#818CF8"),
                        size=160,
                    ),
                    unsafe_allow_html=True,
                )
            with dist_col:
                st.markdown("**Probability Distribution**")
                items = [
                    (CATEGORY_LABELS[c], result["distribution"][c], CATEGORY_COLORS[c])
                    for c in TOP_CATEGORIES
                ]
                signal_bars(items)
        else:
            st.warning("Please enter some text to scan.")
    st.markdown("</div>", unsafe_allow_html=True)

    render_section("Trained Domains")
    domain_cols = st.columns(4)
    domain_descriptions = {
        "cs": "Computer science, machine learning, algorithms & software systems.",
        "math": "Pure and applied mathematics, proofs & modeling.",
        "physics": "General physics research, excluding astrophysics-specific topics.",
        "astro-ph": "Astrophysics, cosmology & observational astronomy.",
    }
    accents = ["cyan", "violet", "amber", "pink"]
    for i, cat in enumerate(TOP_CATEGORIES):
        pod(domain_cols[i], CATEGORY_LABELS[cat], domain_descriptions[cat], accents[i % len(accents)])

    render_section("Model Architecture")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
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
    st.caption("Trained in train_model.py; loaded here via load_weights() for cross-version compatibility.")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.classification_log:
        render_section("Scan History — This Session")
        rows = [
            {
                "dot": "neutral",
                "time": c["timestamp"].strftime("%H:%M:%S"),
                "text": c["text"][:80] + ("..." if len(c["text"]) > 80 else ""),
                "badge": f"{CATEGORY_LABELS.get(c['category'], c['category'])} · {c['confidence']:.0f}%",
            }
            for c in reversed(st.session_state.classification_log)
        ]
        render_terminal_log(rows)


# ============================================================================
# 13. ROUTER
# ============================================================================
ROUTES = {
    "Mission Control": page_mission_control,
    "Archive": page_archive,
    "Signal Analytics": page_signal_analytics,
    "Precision Metrics": page_precision_metrics,
    "Domain Scanner": page_domain_scanner,
}
ROUTES[st.session_state.active_page]()
