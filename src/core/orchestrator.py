# 🧠 The central controller for the RAG flow
from langchain.llms import Ollama
from src.core.components.query_translator import QueryTranslator 
from src.core.components.retriever import  DocumentRetriever
from src.core.components.generator import AnswerGenerator

import os

class RAGOrchestrator:
    """
    Central controller that manages the entire RAG pipeline
    """
    
    def __init__(self):
        """Initialize the orchestrator with all components"""
        # Initialize LLM with Ollama using model "llama3.2"
        self.llm = Ollama(model="llama3.2")

    def process_query(self, query: str) -> str:
        """
        Process a user query through the RAG pipeline

        Step 1: Translate/Decompose the query
        |
        |
        Step 2: Retrieve relevant documents
        |
        |
        Step 3: Generate answer using LLM and retrieved documents

        Args:
            query: User's question or request
            
        Returns:
            Generated answer
        """
        # Step 1: Translate/Decompose the query
        translator = QueryTranslator()
        translated_queries = translator.translate_query(query)
        
        # Step 2: Retrieve relevant documents for each translated query
        retriever = DocumentRetriever()
        docs = retriever.retrieve_documents(translated_queries)
        
        # Step 3: Generate answer using the LLM and retrieved documents
        answer_generator = AnswerGenerator(self.llm)
        answer = answer_generator.generate_answer(docs, query)

        return answer

    def invoke(self, query: str) -> str:
        """Alias for process_query to match Runnable interface"""
        return self.process_query(query)
        

