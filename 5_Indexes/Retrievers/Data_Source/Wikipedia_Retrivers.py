""" It will fetch the data from the *Wikipedia* without any link

## The query need to be right other wise it will not give correct result

It is a retirver that queries the Wikipedia API to fetch relevant content fro a given query
"""
from langchain_community.retrievers import WikipediaRetriever

# Initialize
retriver = WikipediaRetriever(top_k_results=2, lang='en')

# Define the Query
query = "the geopolitical history of India and Pakistan from the perspective of China"

# Get relivent wikipedia document
docs = retriver.invoke(query)

# print(docs)

# print the relevant content
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Source:\n{doc.metadata['source']}")
    print(f"Content:\n{doc.page_content}...")  # truncate for display