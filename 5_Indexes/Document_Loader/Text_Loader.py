"""_Text Loader_
It is a simple and commonly used document loader is langchain that read plain text (.text)
file and covert them into the Lanchain document objects

We have used Cricket.txt for the sample
"""
from langchain_community.document_loaders import TextLoader

loader = TextLoader('Cricket.txt',encoding='utf-8')
# print(loader)
docs = loader.load()
# print(docs)
# print(docs[0])

# It will type
print(type(docs[0]))

# print pageContent
print(docs[0].page_content)
print(docs[0].metadata)