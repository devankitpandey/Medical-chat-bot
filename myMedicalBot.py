import os
import time
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

from langchain.llms import  Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------
load_dotenv()

# -------------------------------------------------------
# App Config
# -------------------------------------------------------
st.title("💬 Medical Bot")
main_placeholder = st.empty()

MODEL_NAME = os.getenv("MODEL_NAME", "nomic-embed-text:latest")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434")
DATA_DIR = Path("data")
INDEX_DIR = Path("model/medical_bot_faiss")  # FAISS saves a folder, not a .pkl
llm = Ollama(base_url=BASE_URL,model=MODEL_NAME)

# -------------------------------------------------------
# Helper: check if FAISS index already exists
# -------------------------------------------------------
def faiss_exists(path: Path) -> bool:
    """Return True if saved FAISS index is already available."""
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()

# -------------------------------------------------------
# Helper: ensure Ollama is reachable
# -------------------------------------------------------
def ensure_ollama_ready(base_url: str):
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(
            f"❌ Cannot reach Ollama at {base_url}. Start Ollama and ensure the model is pulled.\n{e}"
        )
        st.stop()

# -------------------------------------------------------
# Build or load FAISS index
# -------------------------------------------------------
def build_or_load_index():
    embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL)

    if faiss_exists(INDEX_DIR):
        main_placeholder.text("📦 Found existing FAISS index. Loading ...")
        vs = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        main_placeholder.text("🟢 Index loaded successfully.")
        return vs

    # ---- Build index for the first time ----
    main_placeholder.text("📚 No existing index found. Reading PDFs ...")

    if not DATA_DIR.exists():
        st.error(f"Data folder not found: {DATA_DIR.resolve()}")
        st.stop()

    loader = DirectoryLoader(
        path=str(DATA_DIR),
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,  # Use PyPDFLoader if PyMuPDF unavailable
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        st.error("No PDF files found in the /data directory.")
        st.stop()

    # ---- Chunk documents ----
    main_placeholder.text(f"✂️ Splitting {len(docs)} documents into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", "."],
        keep_separator=False,
    )
    chunks = splitter.split_documents(docs)
    total_chunks = len(chunks)
    main_placeholder.text(f"📄 Created {total_chunks} chunks. Generating embeddings ...")

    # ---- Create FAISS index ----
    batch_size = 256
    progress = st.progress(0, text=f"Embedding 0 / {total_chunks}")
    status = st.empty()
    start = time.time()
    vs = None
    first = True

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        if first:
            vs = FAISS.from_documents(batch, embeddings)
            first = False
        else:
            vs.add_documents(batch, embedding=embeddings)
        done = min(i + batch_size, total_chunks)
        progress.progress(done / total_chunks, text=f"Embedding {done} / {total_chunks}")

    elapsed = time.time() - start
    status.info(f"✅ Embeddings complete in {elapsed:.1f}s. Saving index ...")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    progress.empty()
    status.empty()
    main_placeholder.text("🟢 Index built and saved. Future runs will reuse it.")
    return vs

# -------------------------------------------------------
# App Startup
# -------------------------------------------------------
ensure_ollama_ready(BASE_URL)
_ = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL).embed_query("health check")
vectorstore = build_or_load_index()
retriver = vectorstore.as_retriever(search_kwargs={"k":4})
chat_template = ChatPromptTemplate.from_messages([
    ("system","You are an expert medical assistance. Answer fro given context. Be consice and accurate."
              "You need to understand symptoms that user is telling from context you to suggest him possible disease and remedy "),
    ("user","Qestion : {question} \n\n"
            "Context : {context}")
])

def format_doc(docs):
    return "\n\n".join(f"[Chunk {i+1}] {d.page_content}" for i, d in enumerate(docs))

rag_chain = (
    {
        "context": retriver | format_doc,
        "question": RunnablePassthrough(),
    }
    | chat_template
    | llm
    | StrOutputParser()
)

# -------------------------------------------------------
# Chat UI setup
# -------------------------------------------------------
user_messages = [
    {"id": str(uuid.uuid4()), "role": "assistant", "msg": "Hello 👋, how can I help you today?"}
]

if "messages" not in st.session_state:
    st.session_state.messages = user_messages

def update_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["msg"])

update_chat_messages()

# -------------------------------------------------------
# Chat logic
# -------------------------------------------------------
user_message = st.chat_input("Enter your medical query...")

if user_message:
    key_id = str(uuid.uuid4())
    st.session_state.messages.append({"id": key_id, "role": "user", "msg": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        if vectorstore is None:
            st.error("Vector store not ready.")
        else:
            answer = rag_chain.invoke(user_message)
            st.markdown(answer)