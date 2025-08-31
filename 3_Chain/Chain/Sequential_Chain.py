"""_Problem Statment_
1. This the demonstate the sequential chain
2. Here we are taking topic from user then we are genearting detailed Report of that
3. Then sending that detailed Report to llm again to generate the pointer summary.
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

# Defining Prompt1
prompt1 = PromptTemplate(
    template="Generate the details report in this {topic}",
    input_variablees=["topic"],
)
# Defining Pompt 2
prompt2 = PromptTemplate(
    template="Generate the pointer summary for the follwing {text}",
    input_variables=['text']
)

# Defining Parser
parser = StrOutputParser()

# Defining the sequential Chain

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic":"India"}) 
print(result)

# Showing in the form of graph
chain.get_graph().print_ascii()  