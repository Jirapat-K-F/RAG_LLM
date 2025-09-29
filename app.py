import streamlit as st
import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain_core.runnables import RunnablePassthrough
import config  # Activates loading of environment variables
from src.indexing.build_vector_store import VectorIndexBuilder 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build QA Chain
def build_qa_chain(vector_db: VectorIndexBuilder):
    retriever = vector_db.retrieve()
    llm = Ollama(model="llama3.2")
    prompt = hub.pull("rlm/rag-prompt")
    # Chain using Runnable API
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Streamlit App
st.title("RAG Chatbot with FAISS and LLaMA")
st.write("Upload a PDF and ask questions based on its content.")


uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")


if uploaded_file is not None:
    pdf_path = f"/res/uploaded/{uploaded_file.name}"
    os.makedirs("uploaded", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    vector_db = VectorIndexBuilder()
    # text = extract_text_from_pdf(pdf_path)

    st.info("Creating Chroma vector store...")
    vector_db.add_documents(pdf_path)

    st.info("Initializing chatbot...")
    rag_chain = build_qa_chain(vector_db)
    st.success("Chatbot is ready!")


if 'rag_chain' in locals():
    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        st.info("Querying the document...")
        # You can pass a dict with 'question' for tracing
        answer = rag_chain.invoke(question)
        st.success(f"Answer: {answer}")
