"""_Length Based(CharacterTextSplitter)_
It seprated the text based on the length based on chunk size
"""
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Loading Document pdf

loader = PyPDFLoader('Autoregressive_Integrated_Moving_Average_Model_based_Prediction_of_Bitcoin_Close_Price.pdf')
docs = loader.load()

# print(docs[0].page_content)

# Splitting the Text
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

result = splitter.split_documents(docs)

# print(len(result))
print(result[0].page_content)
