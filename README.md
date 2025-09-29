# RAG_LLM

> A local chatbot powered by Retrieval-Augmented Generation (RAG) using Ollama, LangChain, and Streamlit.

---

## 🚀 Features
- **RAG Method**: Combines LLMs with document retrieval for smarter answers.
- **Ollama Integration**: Runs Llama 3.2 locally for privacy and speed.
- **LangChain Community**: Modular pipeline for LLM applications.
- **Streamlit UI**: Simple, interactive web interface.

---

## 🛠️ Setup Instructions

### 1. Install Ollama & Llama 3.2

1. **Download Ollama**
    - Visit [Ollama Downloads](https://ollama.com/download) and install for your OS.
    - Ensure Ollama is running.

2. **Pull the Llama 3.2 Model**
    - Open your terminal and run:
      ```
      ollama pull llama3.2
      ```
    - This downloads the Llama 3.2 model for local use.

---

### 2. Install Python Dependencies

> (Optional) Create a virtual environment for isolation.

Install required libraries:
```
pip install streamlit PyPDF2 langchain sentence-transformers faiss-cpu ollama
```

**GPU Users:**
```
pip install faiss-gpu
```

Update LangChain Community (recommended):
```
pip install -U langchain-community
```

---

### 3. Launch the Chatbot

start virtual env
".\.venv\Scripts\Activate"                                                         

Start the Streamlit app:
```
streamlit run app.py
```

The chatbot will be available at [localhost:8501](http://localhost:8501).

---

Enjoy your private, local LLM chatbot powered by RAG!

