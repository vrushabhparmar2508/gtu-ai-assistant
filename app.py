import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

# --- 1. Setup ---
st.title("🎓 GTU-Assistant (RAG Agent)")
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Please add GOOGLE_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- 2. The Ingestion Layer ---
# For now, put your PDF file in the repo and name it 'notes.pdf'
pdf_path = "notes.pdf" 

if os.path.exists(pdf_path):
    st.write("Reading your GTU notes...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks (essential for RAG)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    # Create Embeddings and Vector Store (The "Brain")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    db = Chroma.from_documents(texts, embeddings)
    
    # --- 3. The Retrieval Layer ---
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    
    # --- 4. User Interface ---
    query = st.text_input("Ask a question about your notes:")
    if query:
        with st.spinner("Agent is searching your notes..."):
            response = qa_chain.invoke(query)
            st.write("### Answer:")
            st.write(response["result"])
else:
    st.warning("Please upload a file named 'notes.pdf' to your repository to start.")
