"""_WebBasedLoader_
It is type of loader which extrac text content directly form the web page(URL).
It uses Beautifulsoup under hood to pass HTML and extract visible text.
Works best for the statistc page.

What is problem statment
1. We are loading data from the Wikipidea
2. We are asking question from the user
3. We passing to LLM to generate the summary out of this
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
    # huggingfacehub_api_token=hf_token  # <--- PASS YOUR TOKEN HERE
)

model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template="""Answer the follwing question \n
    {question}from the following text\n
    {text}
    """,
    input_variables=['question','text']
)

parser = StrOutputParser()

url = "https://en.wikipedia.org/wiki/Perplexity_AI"

loader = WebBaseLoader(url)
docs = loader.load()
# print(docs)

# Passint to llm 

chain = prompt | model | parser

text_1 = "What is the topic talking all bout"

result = chain.invoke({'question': text_1, 'text': docs[0].page_content})
print(result)

