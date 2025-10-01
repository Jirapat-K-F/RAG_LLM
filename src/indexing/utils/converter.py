# this class is responsible for converting raw data into a format suitable for indexing
# eg. text to json
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class Converter:

    def __init__(self, llm):
        map_prompt_template = """
        You are an expert at analyzing incident reports. You are currently viewing a single chunk of a much larger document.
        Extract any details you can find related to the event's cause, condition, consequence, or corrective measures.

        It is OK if some fields are empty, as this is only a partial view of the full report.

        Please respond in the following JSON format:
        {{
            "event_full": "A full, detailed description of the incident from the document.",
            "event_summary": "A concise summary of the event's cause, condition, and consequence.",
            "measures": ["List of specific corrective actions or measures taken"]
        }}

        Content of the document chunk:
        ```{chunk_text}```
        """
        self.llm = llm
        self.map_prompt = ChatPromptTemplate.from_template(map_prompt_template)

    def preprocess(self, raw_data : List[str]) -> List[Document]:
        # Implement your data preprocessing logic here
        print("Running the 'Map' stage on all chunks...")
        reports = []
        
        for i, chunk in enumerate(raw_data):
            try:
                # For now, let's skip the LLM processing to avoid errors
                # and just create documents directly from the raw chunks
                print(f"Processing chunk {i+1}/{len(raw_data)}")
                
                # Create a simple document with the raw chunk content
                partial_report = Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": i,
                        "source": "uploaded_document",
                        "chunk_length": len(chunk)
                    }
                )
                reports.append(partial_report)
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                # Fallback: use raw chunk as content with minimal metadata
                partial_report = Document(
                    page_content=chunk if chunk else "Empty chunk",
                    metadata={"chunk_id": i, "error": str(e)}
                )
                reports.append(partial_report)

        print(f"Successfully processed {len(reports)} chunks")
        return reports
    
class EventReport(BaseModel):
    """A structured report of an event, including its summary and corrective measures."""
    event_full: str = Field(description="A full, detailed description of the incident from the document.")
    event_summary: str = Field(description="A concise summary of the event's cause (สาเหตุ), condition (สภาพแวดrums), and consequence (ผลที่เกิดขึ้น).")
    measures: List[str] = Field(description="A list of specific, numbered corrective actions or measures taken in response to the event.")