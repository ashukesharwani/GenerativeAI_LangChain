"""_Maximal Marginal Relevance(MMR)_
When document has a similar content it will return the reduantant value, 
beacure of sementic search
if same content with differenty way of writting it may give reduantant values
"""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Creating Embedding Model
hf_token = os.getenv("HF_TOKEN")

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction"
)


# Step3: Create Chroma vectore store in memory
VectoreStore = FAISS.from_documents(
    documents=docs,
    embedding=embedding
)

# Enble MMR in the retrivere
retriver = VectoreStore.as_retriever(
    search_type="mmr", #This enable MMR
    search_kwargs={"k": 2, "lambda_mult": 0.5}  
    # k = top results, lambda_mult = relevance-diversity balance
)

query = "What is langchain"

result = retriver.invoke(query)

# print(result)
for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display