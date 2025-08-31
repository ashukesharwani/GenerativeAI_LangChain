"""_summary_
Simple TypeDict practice
**** Parameters
from typing import TypedDict, Annotated,Optional, Literal
1. Annotated - To provide description of the field
2. Annotated[Optional]- This is iptional this value can be there can't not be there
3. Annotated[Literal]- we can restic llm to send result based on our requiremet like for positive we want pos and for negative we want to show neg
"""
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name': "rakesh", 'age': 34}

print(new_person)