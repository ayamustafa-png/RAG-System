import sys

# --- 0. SQLite3 Workaround (required on Streamlit Community Cloud for ChromaDB) ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import json
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

st.set_page_config(page_title="Smart Academic Assistant", layout="wide")
st.title("📚 Smart Academic Research Assistant (Hybrid RAG + DL Framework)")
st.write(
    "An enterprise-grade orchestration combining a Deep Learning Intent Classifier "
    "with a Local Vector DB RAG pipeline."
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


# --- 4. Sidebar Controller Dashboard ---
with st.sidebar:
    st.header("⚙️ System Control Panel")
    st.markdown(
        f"**Instructions:**\n1. Populate `{PAPERS_DIR}/` with source PDF documents.\n"
        "2. Trigger the ingestion pipeline below."
    )

    if st.button("🔄 Execute Ingestion Pipeline"):
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

# Initialize Vector DB link if data exists
vector_db = None
if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- 5. Query Processing & Multi-Model Inference Logic ---
user_query = st.text_input("Input your research question or paste a manuscript abstract here:")

if user_query:
    # Execution Path A: Deep Learning Structural Intent Classification
    sequence = tokenizer.texts_to_sequences([user_query])
    padded_sequence = pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )

    prediction = dl_model.predict(padded_sequence)
    predicted_class = TOP_CATEGORIES[np.argmax(prediction)]
    confidence_score = np.max(prediction) * 100

    st.markdown("### 📊 Deep Learning Text Classifier Analytics:")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Predicted Academic Domain (DL Inference)", value=predicted_class.upper())
    with metric_col2:
        st.metric(label="Classifier Confidence Level", value=f"{confidence_score:.2f}%")

    st.markdown("---")

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
                execution_response = rag_orchestration_chain.invoke({"input": user_query})

                st.subheader("✍️ Generated Synthesized Response:")
                st.write(execution_response["answer"])

                with st.expander("📄 Verifiable Source Citations and Extracted Context:"):
                    for index, document in enumerate(execution_response["context"]):
                        file_origin = os.path.basename(
                            document.metadata.get("source", "Unknown_Reference.pdf")
                        )
                        page_location = document.metadata.get("page", "N/A")
                        st.markdown(
                            f"**Source Document [{index + 1}]:** {file_origin} — "
                            f"(Page Reference: {page_location})"
                        )
                        st.info(document.page_content)
            except Exception as e:
                st.error(
                    f"حصل خطأ أثناء توليد الإجابة من الـ LLM: {e}\n\n"
                    "لو الرسالة بتقول إن الموديل مش متاح (not supported / 404)، غيّري "
                    "قيمة `repo_id` في app.py لموديل تاني متاح دلوقتي على "
                    "https://huggingface.co/models?inference_provider=all&pipeline_tag=text-generation"
                )
    else:
        st.warning(
            "System Notice: RAG pipeline is offline. Populate the target folder and execute "
            "the Vector Ingestion pipeline in the Control Panel."
        )
