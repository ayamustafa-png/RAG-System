import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub  
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. System Environment Architecture ---
PAPERS_DIR = "papers_to_chat"
DB_DIR = "chroma_db"
TOP_CATEGORIES = ['cs', 'math', 'physics', 'astro-ph']
MODEL_PATH = "academic_classifier_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"

if not os.path.exists(PAPERS_DIR):
    os.makedirs(PAPERS_DIR)

st.set_page_config(page_title="Smart Academic Assistant", layout="wide")
st.title("📚 Smart Academic Research Assistant (Hybrid RAG + DL Framework)")
st.write("An enterprise-grade orchestration combining a Deep Learning Intent Classifier with a Local Vector DB RAG pipeline.")

# --- 2. Caching Structural Resource Allocation ---
@st.cache_resource
def initialize_system_resources():
    import pickle
    import os
    import tensorflow as tf
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceHub
    
    # 1. تحديد المسارات بشكل مطلق ومضمون
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_FILE = os.path.join(BASE_DIR, "academic_classifier_model.h5")
    TOKENIZER_FILE = os.path.join(BASE_DIR, "tokenizer.pickle")

    # 2. بناء الموديل بهيكل صريح (بدون batch_shape)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(200,)),
        tf.keras.layers.Embedding(15000, 64),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    # 3. تحميل الأوزان فقط
    model.load_weights(MODEL_FILE)
    # 4. تحميل التوكنيزر
    with open(TOKENIZER_FILE, "rb") as handle:
        token_generator = pickle.load(handle)
    # 5. باقي الخدمات
    embedding_client = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm_node = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
        huggingfacehub_api_token=st.secrets["HF_TOKEN"]
    )
    
    return model, token_generator, embedding_client, llm_node
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    dl_model, tokenizer, embeddings, llm = initialize_system_resources()
else:
    st.error(f"Critical System Failure: Execution assets '{MODEL_PATH}' or '{TOKENIZER_PATH}' are missing. Run 'train_model.py' first.")
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
    st.markdown(f"**Instructions:**\n1. Populate `{PAPERS_DIR}/` with source PDF documents.\n2. Trigger the ingestion pipeline below.")
    
    if st.button("🔄 Execute Ingestion Pipeline"):
        execute_vector_ingestion()
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 📊 Infrastructure Specifications:")
    st.info("• DL Intent Engine: **LSTM (Keras Backend)**\n• Vector DB Hub: **ChromaDB Target**\n• Embedding Model: **All-MiniLM-L6-v2 (HuggingFace)**\n• Core Generative LLM: **Ollama (Llama 3)**")

# Initialize Vector DB link if data exists
vector_db = None
if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- 5. Query Processing & Multi-Model Inference Logic ---
user_query = st.text_input("Input your research question or paste a manuscript abstract here:")

if user_query:
    # Execution Path A: Deep Learning Structural Intent Classification
    sequence = tokenizer.texts_to_sequences([user_query])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    
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
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_orchestration_chain = create_retrieval_chain(retriever_node, question_answer_chain)
        
        with st.spinner("Retrieving local semantic knowledge and generating structural response..."):
            execution_response = rag_orchestration_chain.invoke({"input": user_query})
            
            st.subheader("✍️ Generated Synthesized Response:")
            st.write(execution_response["answer"])
            
            # Context Verification and Citations View Component
            with st.expander("📄 Verifiable Source Citations and Extracted Context:"):
                for index, document in enumerate(execution_response["context"]):
                    file_origin = os.path.basename(document.metadata.get('source', 'Unknown_Reference.pdf'))
                    page_location = document.metadata.get('page', 'N/A')
                    st.markdown(f"**Source Document [{index+1}]:** {file_origin} — (Page Reference: {page_location})")
                    st.info(document.page_content)
    else:
        st.warning("System Notice: RAG pipeline is offline. Populate the target folder and execute the Vector Ingestion pipeline in the Control Panel.")
