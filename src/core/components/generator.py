#5st 
# Generator - Generates the final answer using an LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

class AnswerGenerator:
    """
    Generates final answers using LLM based on retrieved context
    """
    
    def __init__(self, llm):
        """Initialize the answer generator"""
        template = """Answer the question based only on the following context:
            {context}

            Question: {question}
            """
        self.prompt = ChatPromptTemplate.from_template(template)
        self.llm = llm

    def generate_answer(self, retriever : str, query: str) -> str:
        """
        Generate final answer using LLM
        
        Args:
            query: Original user question
            context: Retrieved and processed context documents
            
        Returns:
            Generated answer
        """
        # Chain using Runnable API
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        ans = rag_chain.invoke(query)
        return ans

    def prepare_context(self, documents: list) -> str:
        """
        Format retrieved documents into context for LLM
        """
        pass
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create the prompt for the LLM
        """
        pass
    
    def post_process_answer(self, raw_answer: str) -> str:
        """
        Clean and format the generated answer
        """
        pass