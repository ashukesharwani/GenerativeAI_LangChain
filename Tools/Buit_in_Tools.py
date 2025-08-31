"""
_Built in Tool_
A built-in tools is a tool that langchain already provide for you it's pre
built , production
ready, and requries minimal or no setup

NO need to writ the function logic just import it and use it
"""
# We are using Duck Duck Go Search

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke('Top new in India Today')

print(result)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args)

# Built-in Tool - Shell Tool

# from langchain_community.tools import ShellTool

# shell_tool = ShellTool()

# results = shell_tool.invoke('ls')

# print(results)