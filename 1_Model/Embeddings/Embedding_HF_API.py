"""_summary_ 
We are using API to key for embedding the query
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os

hf_token = os.getenv("HF_TOKEN")

hf = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token,
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = hf.embed_documents(documents)

# Print the embeddings for each document
for i, doc_embedding in enumerate(result):
    print(f"Embedding for document '{documents[i]}':")
    print(f"  Length: {len(doc_embedding)}") # Will be 384 for all-MiniLM-L6-v2
    print(f"  First 5 values: {doc_embedding[:5]}")
    print("-" * 30)

# If you just want to print the whole list of lists
# print(str(result))