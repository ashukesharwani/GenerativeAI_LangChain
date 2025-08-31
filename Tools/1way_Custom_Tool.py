""" _ Method (using @tool decorator)
A Custom tool is a tool that you define yourself
-> Creat a custom tool for multiplication
"""
from langchain_core.tools import tool


# Step1 Creat a function
def multiply(a, b):
    '''Multiply two number'''
    return a*b


# Step2 add type hint
def multiply(a: int, b: int) -> int:
    '''Multiply two number'''
    return a*b


# Step3 add tool decorator (This is main)
@tool
def multiply(a: int, b: int) -> int:
    '''Multiply two number'''
    return a*b


result = multiply.invoke({'a': 3, 'b': 3})
print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)

# What llm receive when we sent to him
print(multiply.args_schema.model_json_schema())