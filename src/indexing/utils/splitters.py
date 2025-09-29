# Custom text splitters (e.g., Semantic Splitter)

from typing import List, Dict, Any, Optional
import re

import os
from typing import List, Dict, Any
import streamlit as st
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
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
class TextSplitter:
    """
    Splits text based on semantic boundaries rather than fixed chunk sizes
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.8):
        """
        Initialize the semantic splitter
        
        Args:
            model_name: Embedding model for similarity calculation
            similarity_threshold: Threshold for semantic similarity
        """
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def chunking_text(self, text: str) -> List[str]:
        """
        Split text based on semantic similarity
        
        Args:
            text: Input text to split
            
        Returns:
            List of semantically coherent text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splitted_text = text_splitter.split_documents(text)
        return splitted_text

    def calculate_sentence_similarities(self, sentences: List[str]) -> List[float]:
        """
        Calculate similarities between consecutive sentences
        """
        pass
    
    def find_split_points(self, similarities: List[float]) -> List[int]:
        """
        Find optimal split points based on similarity scores
        """
        pass

