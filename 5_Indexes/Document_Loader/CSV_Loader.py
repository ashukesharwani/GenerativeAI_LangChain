"""_CSV_Loader_
It is a document loader used to load CSV files into langChain Document objects one per row , by default
-> It create a document_object for every row
"""
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Titanic Dataset.csv')
docs = loader.load()

print(docs[0].page_content)