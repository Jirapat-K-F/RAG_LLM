# 🧠 The central controller for the RAG flow
from langchain.llms import Ollama
from core.components.query_translator import QueryTranslator 
from core.components.router import  QueryRouter
from core.components.query_constructor import QueryConstructor
from core.components.retriever import  DocumentRetriever
from core.components.generator import AnswerGenerator

import os

class RAGOrchestrator:
    """
    Central controller that manages the entire RAG pipeline
    """
    
    def __init__(self):
        """Initialize the orchestrator with all components"""
        # Initialize LLM with Ollama using model "llama3.2"
        self.llm = Ollama(model="llama3.2")
        self._setup_langsmith_tracking()

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
        all_retrieved_docs = []
        for tq in translated_queries:
            docs = retriever.retrieve_documents(tq)
            all_retrieved_docs.extend(docs)
        
        # Step 3: Generate answer using the LLM and retrieved documents
        answer_generator = AnswerGenerator(self.llm)
        final_answer = answer_generator.generate_answer(query, all_retrieved_docs)
        
        return final_answer

    def _setup_langsmith_tracking(self):
        """Initialize LangSmith tracing"""
        os.environ.update({
            'LANGCHAIN_TRACING_V2': os.getenv('LANGCHAIN_TRACING_V2'),
            'LANGCHAIN_ENDPOINT': os.getenv('LANGCHAIN_ENDPOINT'),
            'LANGCHAIN_API_KEY': os.getenv('LANGCHAIN_API_KEY'),
            'LANGCHAIN_PROJECT': os.getenv('LANGCHAIN_PROJECT')
        })
        
    def _setup_components(self):
        """Initialize all RAG components"""
        pass


