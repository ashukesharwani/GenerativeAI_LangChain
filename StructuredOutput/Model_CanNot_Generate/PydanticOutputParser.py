"""_Problem_Statement_
# This is PydanticOutpuParser
From given text ask form llm what is it's name,age and city
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
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


# Creating Pydantic model
class Person(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(gt=18, description="Age of Person")#ge is greate then 18 , lt is less then
    city: str = Field(description="city")


parser = PydanticOutputParser(pydantic_object=Person)

# Creating Prompt
template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# Crating Chain
prompt = template.invoke({"place":"India"})
print("This is the prompt\n")
print(prompt)
result = model.invoke(prompt)
print("This is the Result form LLM\n")
print(result)
R = parser.parse(result.content)
print("This is the Result after sending to Pydantic\n")
print(R)


"""This is using Chain"""
# chain=template|model|parser
# final=chain.invoke({"place":"India"})
# print(final)
    