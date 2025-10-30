import streamlit as st
import os
import config  # Activates loading of environment variables
from src.indexing.build_vector_store import VectorIndexBuilder 
from src.core.orchestrator import RAGOrchestrator
from datetime import datetime

# Print a short message to the terminal on every Streamlit rerun so you can see reloads
print(f"[streamlit] reload at {datetime.now().isoformat()} (PID={os.getpid()})")

# Streamlit App
st.title("RAG Chatbot with FAISS and LLaMA")
st.write("Upload a PDF and ask questions based on its content.")

# Display currently uploaded files
st.subheader("Currently Uploaded Files")
uploaded_dir = "res/uploaded"
if os.path.exists(uploaded_dir) and os.listdir(uploaded_dir):
    files = os.listdir(uploaded_dir)
    st.write("📁 Files in database:")
    for file in files:
        file_path = os.path.join(uploaded_dir, file)
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        st.write(f"• {file} ({file_size_mb:.2f} MB)")
else:
    st.write("📂 No files currently uploaded")

st.divider()

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Check if the vector database already exists
vector_db = VectorIndexBuilder()
if vector_db.vectorstore is not None:
    st.info("Vector database found. You can start asking questions.")
    question = st.text_input("Ask a question about the existing database:")
    if question:
        st.info("Querying the database...")
        rag_chain = RAGOrchestrator()
        answer = rag_chain.invoke(question)
        st.success(f"Answer: {answer}")

# Handle file uploads
if uploaded_file is not None:
    pdf_path = f"res/uploaded/{uploaded_file.name}"
    os.makedirs("res/uploaded", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Creating Chroma vector store...")
    vector_db.add_documents(pdf_path)

    st.info("Initializing chatbot...")
    rag_chain = RAGOrchestrator()
    st.success("Chatbot is ready! You can now ask questions.")
    st.rerun()  # Refresh the page to show the new file in the list
    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        st.info("Querying the document...")
        answer = rag_chain.invoke(question)
        st.success(f"Answer: {answer}")

# Add a button to clear the vector database
if st.button("Clear Vector Database"):
    if vector_db.vectorstore is not None:
        vector_db.clear_vectorstore()
        st.success("Vector database and uploaded files cleared successfully.")
        st.rerun()  # Refresh the page to update the file list
    else:
        st.warning("No vector database found to clear.")
