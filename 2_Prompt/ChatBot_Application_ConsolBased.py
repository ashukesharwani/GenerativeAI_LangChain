'''
Multiple-turn Conversation
Storing the data in list
Application Create a ChatBot Consol Based 
'''
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os


load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ", result.content)
print(chat_history)

"""
    The problem in this chatbot is
    we are not able to find which one is the user generated and which is the AI generated
    Becaue of the as chat base will incrase it will be difficult for the llm to understand 
    which one is user generated and which one is the AI generated
 """