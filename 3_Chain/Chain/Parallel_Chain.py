"""_Problem_Statement_
from given summary 
1. Create notes
2. Create Quize
Then merge and show to student

how to this will exccule
1. parallel chain then this output will got to merge_chain

"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it", 
    task="text-generation",
    # huggingfacehub_api_token=hf_token  # <--- PASS YOUR TOKEN HERE
) 
model = ChatHuggingFace(llm=llm)

# Text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Creating Prompt

# Createing prompt1 for creating notes
prompt1 = PromptTemplate(
    template="Create a notes of a summary {summary}",
    input_variables=['summary']
)
# Createing prompt2 for creating Quize
prompt2 = PromptTemplate(
    template="Create a 5 quize from this summary{summary}",
    input_variables=['summary']
)

# Creating prompt3 to merget both notes and quize
prompt3 = PromptTemplate(
    template=
    """
    Merge the provided notes and quize into a single document\n
    notes->{notes}
    and quize->{quize}
    """,
    input_variable=['notes', 'quize']
    
)

# Creating Parser
parser = StrOutputParser()

# Creating Chain
parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model | parser,
        'quize': prompt2 | model | parser
    }
)
# how to access the any single element
# reslt=parallel_chain.invoke({'summary':text})
# print(reslt['notes'])

# For Merge we have written this
merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'summary':text})
print(result)

chain.get_graph().print_ascii()
