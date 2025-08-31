""" Method_(using Structured Tool & Pydantic)
A structured tool in langchain is a special type of tool where the input to 
the tool follows a structured schema , typically defined using a Pydatic method
"""
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")


def multiply_func(a: int, b: int) -> int:
    return a*b


multiply_tool = StructuredTool.from_function(   
    func=multiply_func,
    name="multiply",
    description=" Multiply two number",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 2, 'b': 4})
print(result)

print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)
