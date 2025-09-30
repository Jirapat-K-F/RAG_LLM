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

        documents_name = os.path.basename(documents_path)
        chunk_ids = self.generateId(documents_name, splitted_doc)

        if self.vectorstore is None:
            print("Vector store not initialized. Creating a new one...")
            self.create_from_documents(splitted_doc, chunk_ids)
            return

        print("Adding new documents to the vector store...")
        
        self.vectorstore.add_documents(documents=splitted_doc, ids=chunk_ids)
        print(f"Successfully added {len(splitted_doc)} new document chunks.")

    def create_from_documents(self, splitted_doc, chunk_ids):
        print(f"Creating vector store in '{self.persist_directory}'...")
        self.vectorstore = Chroma.from_documents(
            documents=splitted_doc,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
            ids=chunk_ids
        )
        print("Vector store created successfully.")

    def retrieve(self, query: str, search_type="similarity") -> List[str]:
        """
        Load and extract text from documents
        """
        if self.vectorstore is None:
            # If the vector store isn't created yet, try to load it from disk
            print(f"Vector store not created. Loading from '{self.persist_directory}'...")
            self._load_vectorstore()
            print("Vector store loaded successfully.")
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": self.K}
        )
        # docs = retriever.get_relevant_documents(query)
        return retriever

    def update_documents(self, old_documents_path: str, new_documents_path: str):
        """
        Updates existing documents in the vector store.
        """
        if self.vectorstore is None:
            print("Vector store not initialized. Cannot update documents.")
            return
        print(f"Updating documents from '{old_documents_path}'... to '{new_documents_path}'")
        self.delete_documents(old_documents_path)
        self.add_documents(new_documents_path)

    def delete_documents(self, doc_path: str):
        """
        Deletes documents from the vector store by their IDs.
        """
        if self.vectorstore is None:
            print("Vector store not initialized. Cannot delete documents.")
            return
        doc_id = self.retrieveIds(doc_path)
        print(f"Deleting documents with IDs: {doc_ids}")
        self.vectorstore.delete(doc_ids)
        print("Documents deleted successfully.")    

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

    def generateId(source_id, chunks):
        # Generate custom IDs and add shared metadata
        chunk_ids = [f"{source_id}-{i}" for i, _ in enumerate(chunks)]
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_id"] = source_id

    def retrieveIds(self, doc_path: str) -> str:
        """
        Generates document main IDs based on the document path.
        """
        # Assuming 'vectorstore' is your loaded Chroma object
        print("Retrieving all documents from the vector store...")
        base_id = os.path.basename(doc_path)
        # The .get() method retrieves documents, IDs, and metadata
        retrieved_data = [id for id in self.vectorstore.get()['ids'] if base_id in id] # No arguments needed to get everything

        return retrieved_data