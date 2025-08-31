from langchain_core.prompts import ChatPromptTemplate

# Define the ChatPromptTemplate
chat_template = ChatPromptTemplate(
    [
        ('system', 'You are the helpful {domain} expert'),
        ('human', 'Explain in simple terms,what is the {topic}')
    ]
)

prompt = chat_template.invoke({'domain': 'Cricket', 'topic': 'what is superover'})
print(prompt)
# https://github.com/campusx-official/langchain-prompts/blob/main/chat_prompt_template.py
