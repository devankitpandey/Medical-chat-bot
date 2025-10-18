import os
import pickle
import time

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader

load_dotenv()
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
import uuid
import streamlit as st
from langchain_community.vectorstores import FAISS
## Design the bot screen -
st.title("💬 Medical Bot")
main_placeholder = st.empty()
embedding_path = "medical_bot_faiss.pkl"
### Do the pre-processing of data
MODEL_NAME = os.getenv("MODEL_NAME","openhermes:latest")
BASE_URL = os.getenv("BASE_URL","http://localhost:11434/")
main_placeholder.text("✅ Loading medical encyclopedia ...")
pdf_loader = DirectoryLoader(loader_cls=PyPDFLoader,path="data",glob="*.pdf")
pdf_loaded_data = pdf_loader.load()
print(f"Loaded {len(pdf_loaded_data)} elements")
print("Sample element:", pdf_loaded_data[0].metadata.get("category"), pdf_loaded_data[0].page_content[:200], "...")

main_placeholder.text(f"✅ Creating Chunks of data #{len(pdf_loaded_data)}...")
txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", ","],   # coarse → fine
    keep_separator=False
)
split_docs = txt_splitter.split_documents(pdf_loaded_data)

main_placeholder.text(f"✅ Creating Embeddings ...")

try:
    r = requests.get(f"{BASE_URL}/api/tags", timeout=5)
    r.raise_for_status()
    main_placeholder.text(f"✅ Connecting model ...")
except Exception as e:
    st.error(f"Cannot reach Ollama at {BASE_URL}. Start Ollama and ensure the model is pulled.\n{e}")
    st.stop()

# --- 1) Construct embeddings (ensure strings) ---
embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL)

# --- 2) Minimal sanity embed (fast fail if model missing) ---
try:
    _ = embeddings.embed_query("health check")
    main_placeholder.text(f"✅ Model connection done , checking embeddings ... ")
except Exception as e:
    st.error(f"Embedding model not ready (pull it with `ollama pull {MODEL_NAME}`):\n{e}")
    st.stop()


embeddings = OllamaEmbeddings(model=MODEL_NAME)
batch_size = 50
total_docs = len(split_docs)
progress = st.progress(0,text=f"Embeddings 0/{total_docs}")
status = st.empty()
ollama_embeding = None
start = time.time()
main_placeholder.text(f"✅ Processing embeddings ... ")
for i in range(0,total_docs,batch_size):
    batch = split_docs[i:i+batch_size]
    if ollama_embeding is None:
        ollama_embeding = FAISS.from_documents(documents=batch,embedding=embeddings)
    else:
        ollama_embeding.add_documents(batch,embeddings=embeddings)
    done = min(i + batch_size, total_docs)
    progress.progress(done / total_docs, text=f"Embedding {done} / {total_docs}")

    elapsed = time.time() - start
    status.info(f"Embeddings complete in {elapsed:.1f}s. Saving index...")

    # --- 5) Save index ---
os.makedirs(embedding_path, exist_ok=True)
ollama_embeding.save_local(embedding_path)
progress.empty()
status.empty()
main_placeholder.text("🟢 Ready !!")


user_messages = [{"id": str(uuid.uuid4()),"role":"assistant","msg":"Hello , How I can help you ?"}]

if "messages" not in st.session_state:
    st.session_state.messages = user_messages

## Actions

def update_chat_messages():
    messages_from_session = st.session_state.messages
    for message in messages_from_session:
       with st.chat_message(message["role"]):
           st.markdown(message["msg"])





update_chat_messages()
user_message = st.chat_input("Enter your query...")
if user_message:
    key_id = uuid.uuid4()
    st.session_state.messages.append({"id": str(key_id),"role":"user","msg":user_message})
    with st.chat_message("user"):
        st.markdown(user_message)







