from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# 1. Load environment variables
load_dotenv()

# 2. Access your Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Check if the token is loaded (good for debugging)
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file.")
else:
    print("Hugging Face token loaded successfully.")


# 3. Initialize HuggingFaceEndpoint, passing the hf_token
#    The 'huggingfacehub_api_token' parameter is used to pass the token.
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token  # <--- PASS YOUR TOKEN HERE
)

# 4. Initialize ChatHuggingFace with the llm
model = ChatHuggingFace(llm=llm)

print("ChatHuggingFace model initialized:")
print(model)

# 5. Invoke the model
print("\nInvoking the model...")
result = model.invoke("What is the capital of India")

print("\nResult:")
print(result.content)