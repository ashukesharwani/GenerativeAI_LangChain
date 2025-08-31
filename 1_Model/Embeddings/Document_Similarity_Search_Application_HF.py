'''
This is using HuggingFace API
'''

from langchain_huggingface import HuggingFaceEndpointEmbeddings
#This for the find the similarity in text
from sklearn.metrics.pairwise import cosine_similarity 
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import os

# hf_token = os.getenv("HF_TOKEN")

model_embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction",
    # huggingfacehub_api_token=hf_token,
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# User Query
query = 'tell me about cam crickter batter'

# Creatign embedding for the document
document_embed = model_embedding.embed_documents(documents)
print(np.array(document_embed).shape)

# Creating embedding for the query
query_embed = model_embedding.embed_documents(query)
# print(query_embed)
print(np.array(query_embed).shape)

# Reshping the error expected 2 day arrya
A = np.array(document_embed)   # shape (5, 768)
B = np.array(query_embed)  # shape (20, 768)

# Finding Similarity
scores = cosine_similarity(B,A)[0]
# print(scores)
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)



