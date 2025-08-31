"""_Problem_Statement_
This is for the Conditional_Chain
user->Feedback(Analysis -ve and +ve)
1. For +ve reply thankyou
2. For -ve reply sorry
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
# import pydantic libraries
from pydantic import BaseModel, Field
from typing import Literal
# import the Runnable for condition Chain
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableLambda

from dotenv import load_dotenv
import os

# Defing model and call LLM
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it", 
    task="text-generation",
    # huggingfacehub_api_token=hf_token  # <--- PASS YOUR TOKEN HERE
) 
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# Creating PydanticOutput Parser
class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# print(classifier_chain.invoke({'feedback': 'This is a beautiful phone'}))

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
"""_Syntx Conditional Chain_
branch_chain = RunnableBranch(
    (Condition1,Chain),
    (Condition2,Chain),
    Default Chain   
)
"""

branch_chain = RunnableBranch(
    # syntx = (Condition1,Chain)
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a beautiful phone'})
print(result)

chain.get_graph().print_ascii()

