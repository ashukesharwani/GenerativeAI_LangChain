"""_Text_Structured_Based(RecursiveCharacterTextSplitter)_
It separates the text based on structure pagragraph , sentences, words, characters
This is the structure
1. \n\n - paragrah
2. \n - senteces
3. "_" - word
4. '.' - Character
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""
# Initialize the splitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)

# Perform the splitting
chunk = splitter.split_text(text)

print(len(chunk))
print(chunk[0])
