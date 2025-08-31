"""
1. Single Turn Stand alone query Dynamic_prompt
2. Using Prompt Template to define the prompt - for single turn stamd alone query.

"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token  # <--- PASS YOUR TOKEN HERE
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

# Selection 1
paper_input = st.selectbox(
    "Select Research paper Name",
    ["Generative Adversarial Networks", "Attention Is All You Need", "Denoising Diffusion Probabilistic Models"]
)

# Selection 2
style_input = st.selectbox(
    "Explanation Style",
    ["short Paragraph", "Diagramatically", "Mathmatically"]
)

# Selection 2
length_input = st.selectbox(
    "Select Lenght",
    ["Short", "Long", "Small"]
)

# Creating Template

template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}" 
    with the following specification explanation style : {style_input}
    Explanation style :{length_input}
    1. Mathematical Details
    2. Analogies
    """,
    input_variables=["paper_input", "style_input", "lenght_input"]
)

prompt = template.invoke(
    {"paper_input": paper_input,
     "style_input": style_input,
     "length_input": length_input}
)

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)

# how to run this
# streamlit run "file.py"