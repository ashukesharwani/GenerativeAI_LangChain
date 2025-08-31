'''
Multiple-turn Conversation Including Message Type
1. Adding system Message for distinguish between AI message, user message, system message
2. Using "ChatPromotTemplate" - it for multi turn conversation
2. Storing the chat history we are using "Message Placeholder"
'''
"""_summary_
    In this we are storing the chat history of the customer in the file name chat_history.py
    while user is talking to bot we are giving  him answer based on old history.
"""
   
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

print(f"\n***********************\n")
# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})

print(prompt)