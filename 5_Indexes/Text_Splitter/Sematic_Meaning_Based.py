"""_Based_ON Sementic Meaning_
How it woks
First it embed the each sentences then try to match each sentences if found it similarity is lwot then make a chunk

"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os

hf_token = os.getenv("HF_TOKEN")

hf = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction",
    # huggingfacehub_api_token=hf_token,
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

text_splitter = SemanticChunker(
    hf, 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs[0])