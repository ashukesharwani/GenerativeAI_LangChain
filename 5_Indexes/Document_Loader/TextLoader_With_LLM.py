"""_Text Loader (Problem Statemnt)_
We need to send the document to llm to generate summary of the poem
poem is in the text file"Cricet.txt"
"""
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
    # huggingfacehub_api_token=hf_token  # <--- PASS YOUR TOKEN HERE
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# Creating Prompt
prompt = PromptTemplate(
    template="Write a summary for the following poem \n {poem}",
    input_variables=["poem"]
)
# loading Document
loader = TextLoader('Cricket.txt', encoding='utf=8')
docs = loader.load()
# print(type(docs[0]))

chain = prompt | model | parser

result = chain.invoke({'poem': docs}) 
print(result)

