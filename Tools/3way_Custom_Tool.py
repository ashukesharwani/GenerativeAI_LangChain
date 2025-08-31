""" Method_(using Base Tool Class)
BaseTool is the abstract base class for tall tools in langchain.
All other tools type are bulit in the top of base tool
"""

from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


# Arg scheam using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")


class MultiplyTool(BaseTool):
    name: str = "Multiply"
    description: str = "Multiply two number"    
    args_schema: Type[BaseModel] = MultiplyInput
# add _ in run in mendatory other wise it wil show error

    def _run(self, a: int, b: int) -> int:
        return a*b


multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a': 4, 'b': 6})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)

print(multiply_tool.args)
