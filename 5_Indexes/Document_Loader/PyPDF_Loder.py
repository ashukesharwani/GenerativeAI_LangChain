"""_PyPDF_Loader_
-> It is used to load the content from PDF file and convert each page into a Document object
-> EX- every page int 1 document object. 100 page means 100 document object
-> There are so many different type of pdf loader based on uses
"""
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Social_Media_Message_impact_on_Crypto.pdf")
docs = loader.load()

for idx, doc in enumerate(docs, start=1):
    print(f"Document {idx} - Page {doc.metadata.get('page_number', 'unknown')}")
    # print(doc.metadata.get('author', 'unknown'))  

# print(len(docs))
print(docs[0])