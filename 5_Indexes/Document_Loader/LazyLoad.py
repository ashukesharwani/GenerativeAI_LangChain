"""_Load Vs LazyLoad_
load-> it loads everything at once
Lazy_load-> loads on demand
"""

from langchain_community.document_loaders import TextLoader
# loading Document
loader = TextLoader('Cricket.txt', encoding='utf=8')
docs = list(loader.lazy_load())
print(docs[0].metadata)

# print(result)