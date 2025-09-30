# this class is responsible for converting raw data into a format suitable for indexing
# eg. text to json
class Converter:

    def __init__(self, llm):
        self.llm = llm

    def preprocess(self, raw_data):
        # Implement your data preprocessing logic here
        processed_data = raw_data  # Placeholder for actual processing
        return processed_data