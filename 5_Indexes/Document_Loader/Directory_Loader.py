"""
It is a document loader that let you load multiple document form a folder of file
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "C:/Users/HP/OneDrive/Desktop/Resarach paper",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)
docs = loader.load()

for idx, doc in enumerate(docs, start=1):
    print(f"Document {idx} - Page {doc.metadata.get('page_number', 'unknown')}")