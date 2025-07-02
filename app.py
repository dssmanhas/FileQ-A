import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import tempfile, os, hashlib
import torch
from langchain_community.vectorstores import FAISS
# Define directories
UPLOAD_DIR = "uploaded_docs"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Load LLM (LaMini-T5)
@st.cache_resource
def llm_pipeline():
    checkpoint = "MBZUAI/LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="cpu", torch_dtype=torch.float32)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256, do_sample=True, temperature=0.3, top_p=0.95)
    return HuggingFacePipeline(pipeline=pipe)

# Save uploaded file
def save_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

# Generate a unique ID for each file (to avoid reprocessing)
def file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

#Load document content
def load_document(file_path):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return UnstructuredWordDocumentLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    else:
        return []
# def load_document(file_path):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File does not exist: {file_path}")
#     if os.path.getsize(file_path) == 0:
#         raise ValueError(f"The file is empty: {file_path}")
#     return PyPDFLoader(file_path).load()
def main():
    st.set_page_config(page_title="📚 Chat with Documents", layout="wide")
    st.title("📄 Chat with PDF / DOCX / TXT Files")

    uploaded_files = st.file_uploader("📎 Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        doc_hashes = []
        file_paths = []

        # Save each uploaded file once and store its path
        for uploaded_file in uploaded_files:
            file_path = save_file(uploaded_file)
            file_paths.append(file_path)
            doc_id = file_hash(file_path)
            doc_hashes.append(doc_id)

            # Check if already embedded
            doc_db_path = os.path.join(CHROMA_DIR, doc_id)
            if os.path.exists(doc_db_path):
                st.info(f"✅ {uploaded_file.name} already indexed. Loading from DB.")
            else:
                st.info(f"📄 Processing {uploaded_file.name}...")
                docs = load_document(file_path)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                st.info(f"📌 Storing vectors for {uploaded_file.name}...")
                vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=doc_db_path)
                vectordb.persist()

        # Now process all files using the stored file paths
        all_chunks = []
        for file_path in file_paths:
            docs = load_document(file_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

       # vectordb = Chroma.from_documents(all_chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
        vectordb = FAISS.from_documents(all_chunks, embedding=embeddings)
        vectordb.persist()
        retriever = vectordb.as_retriever()
        llm = llm_pipeline()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )

        st.success("✅ Ready! Ask questions below.")

        # Initialize chat history state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history in order
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)

        # Input box at the bottom like real chatbot
        user_input = st.chat_input("💬 Ask a question about your documents...")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            # Combine last 3 chat turns as context
            context = ""
            for q, a in st.session_state.chat_history[-3:]:
                context += f"User: {q}\nAssistant: {a}\n"
            context += f"User: {user_input}\n"

            with st.spinner("🤖 Generating answer..."):
                result = qa_chain.invoke(context)
                answer = result["result"]

            # Display bot reply
            with st.chat_message("assistant"):
                st.markdown(answer)

    # Store exchange in session
            st.session_state.chat_history.append((user_input, answer))
if __name__ == "__main__":
    main()