from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit import text

class Reader:

    def __init__(self, llm):
        self.llm = llm

        # Translation of the prompt:
        # "You are an expert in the Thai language. Please correct the following sentence to be grammatically correct and fix any spelling errors.
        # Incorrect sentence: '{input_text}'
        # Please respond with only the corrected sentence."
        self.prompt_template_str = """คุณคือผู้เชี่ยวชาญด้านภาษาไทย โปรดแก้ไขประโยคต่อไปนี้ให้ถูกต้องตามหลักไวยากรณ์และแก้ไขคำที่สะกดผิด
        ประโยคที่ผิด: "{input_text}"
        กรุณาตอบเฉพาะประโยคที่แก้ไขแล้วเท่านั้น"""

    def read(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in (reader.pages):
            text += page.extract_text()
        return text
    
    # this function will read and correct the text
    # it also very slow and cost a lot of tokens
    def advanceRead(self, file_path: str, llm) -> str: 
        text = self.read(file_path)
        advance_text = self.correct_text(text) #correct the thai language text
        return advance_text

    def correct_text(self, text: str) -> str:
        # 1. Instantiate the Ollama Chat Model
        prompt = ChatPromptTemplate.from_template(self.prompt_template_str)

        # 3. Define an Output Parser
        # StrOutputParser() simply takes the model's output and converts it into a plain string.
        output_parser = StrOutputParser()

        # 4. Build the Chain using LangChain Expression Language (LCEL)
        # This pipes the components together: the prompt is sent to the model,
        # and the model's output is sent to the parser.
        chain = prompt | self.llm | output_parser

        # --- 5. Invoke the Chain with your input text ---
        # print(f"Sending imperfect text to LangChain: '{text}'")

        try:
            # The input is a dictionary where the key matches the variable in your prompt template.
            corrected_text = chain.invoke({"input_text": text})
            return corrected_text
        except Exception as e:
            print(f"\nAn error occurred!")
            print("Please ensure the Ollama application is running and you have pulled the 'llama3' model.")
            print(f"Error details: {e}")
