"""_About_
1. This code is to show the simple sequential chain working and how it looks by diagram
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    # huggingfacehub_api_token=hf_token  # <--- PASS YOUR TOKEN HERE
)

model = ChatHuggingFace(llm=llm)

# Defining Prompt
prompt = PromptTemplate(
    template="Generate 5 interesting fact about {topic}",
    input_variables=["topic"],
)

# Defining Parser
parser = StrOutputParser()

# Defining Chain
chain = prompt | model | parser

# Providing input in the chain
result = chain.invoke({"topic": "Cricket"})
print(result)

# Showing in the form of graph
chain.get_graph().print_ascii()