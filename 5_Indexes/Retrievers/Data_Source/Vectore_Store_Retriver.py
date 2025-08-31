"""
It will let you search and fetch documents from a vecotore store based on 
semantic similairty using vectore embeddings 

NOtes:  Same thing can be done in Vectore store but what is difference is here 
we can used diffirent search techniques but in vectore store we can only use on search techniques
"""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
    ]
# Setp 3: Initialize embedding Model
hf_token = os.getenv("HF_TOKEN")

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction"
)

# Step3: Create Chroma vectore store in memory
VectoreStore = FAISS.from_documents(
    documents=documents,
    embedding=embedding
    # collection_name="my_collections"
)
# Step4 Convert vector into a retrivers

retrivers = VectoreStore.as_retriever(
    search_kwargs={"k": 2}
)
query = "what is Chroma used for"

result = retrivers.invoke(query)
# print(result)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display