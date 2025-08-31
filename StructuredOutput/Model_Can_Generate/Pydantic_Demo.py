"""_summary_
1. if it's "int" then we can't pass str in that it will show error(It does validation)
2. We can set default value in this 
"""

from pydantic import BaseModel
class Student(BaseModel):
    name:str
    # name:str="Asutosh"  ## To set the default if vlaues is not given
    
new_student={"name":"Rakesh"}

students=Student(**new_student)
print(students.name)