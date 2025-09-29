# 📜 Script to create and save the vector index

import os
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.indexing.utils.reader import Reader
from langchain_community.llms import Ollama
from src.indexing.utils.splitters import TextSplitter

class VectorIndexBuilder:
    """
    Builds and manages vector indices for document retrieval
    """
    def __init__(self):
        """
        Initialize the vector index builder
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.llm = Ollama(
            model="llama3.2",
            temperature=0.3
        )
        self.persist_directory = os.getenv("PERSIST_DIRECTORY")
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        # Load the vector store on initialization if it already exists
        self._load_vectorstore()
        self.reader = Reader(self.llm)
        self.splitter = TextSplitter()
        self.K = 3  # Number of top documents to retrieve

    def add_documents(self, documents_path: str):
        """
        Adds new documents to the existing vector store.
        """
        documents = self.reader.read(documents_path)
        splitted_doc = self.splitter.chunking_text(documents)

        if self.vectorstore is None:
            print("Vector store not initialized. Creating a new one...")
            self.create_from_documents(splitted_doc)
            return

        print("Adding new documents to the vector store...")
        # Split the new documents before adding

        self.vectorstore.add_documents(splitted_doc)
        print(f"Successfully added {len(splitted_doc)} new document chunks.")

    def create_from_documents(self, splitted_doc, chunk_size=1000, chunk_overlap=200):
        print(f"Creating vector store in '{self.persist_directory}'...")
        self.vectorstore = Chroma.from_documents(
            documents=splitted_doc,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        print("Vector store created successfully.")

    def retrieve(self, search_type="similarity") -> List[str]:
        """
        Load and extract text from documents
        """
        if self.vectorstore is None:
            # If the vector store isn't created yet, try to load it from disk
            print(f"Vector store not created. Loading from '{self.persist_directory}'...")
            self._load_vectorstore()
            print("Vector store loaded successfully.")
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": self.K}
        )
    
    def save_index(self, index_path: str) -> bool:
        """
        Save the vector index to disk
        """
        pass
    
    def update_index(self, new_documents: List[str], index_path: str) -> bool:
        """
        Update existing index with new documents
        """
        pass

    def _load_vectorstore(self):
        """Loads the vector store from disk if it exists."""
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from '{self.persist_directory}'...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            print("Vector store loaded successfully.")
        else:
            print("No existing vector store found. A new one will be created.")