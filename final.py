#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
import cassio
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter
from pypdf import PdfReader


# In[20]:


pip install -U langchain-community


# In[21]:


from langchain_community.llms import OpenAI


# In[22]:


# Set up Streamlit page
st.title("Search for query")

# Input your AstraDB and OpenAI credentials
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:dvhmxMiwHeIgjNcOePvQqynw:6e94b5c21fe013849421f626c1f0cdddbc70e5e4909e81f702972d20b3398b3d"
ASTRA_DB_ID ="be688535-2592-4339-9f95-5383184389b3"
OPENAI_API_KEY = "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"
ASTRA_DB_API_ENDPOINT = "https://be688535-2592-4339-9f95-5383184389b3-us-east1.apps.astra.datastax.com"


# In[ ]:


# Read PDF and extract text
pdfreader = PdfReader("48lawsofpower.pdf")
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Initialize AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize OpenAI
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# In[ ]:


# Initialize AstraDB Vector Store
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="my_desired_table",
    session=None,
    keyspace=None,
)

# Split text using Character Text Splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)


# In[ ]:


# Add texts to AstraDB Vector Store
astra_vector_store.add_texts(texts)

# Display number of headlines inserted
st.write(f"Inserted {len(texts)} headlines.")

# Initialize AstraDB Vector Store Index
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)


# In[ ]:


# User interaction loop
first_question = True
while True:
    if first_question:
        query_text = st.text_input("Enter your question (or type 'quit' to exit):", key="question_input").strip()
    else:
        query_text = st.text_input("What's your next question (or type 'quit' to exit):", key="next_question_input").strip()

    # ... (other code)

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False

    st.write(f"\nQUESTION: \"{query_text}\"")
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    st.write(f"ANSWER: \"{answer}\"")

    st.write("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        st.write(f"    [{score:.4f}] \"{doc.page_content[:84]} ...\"")


# In[ ]:


streamlit_version = st.__version__
print(f"Streamlit version: {streamlit_version}")


# In[ ]:





# In[ ]:





# In[ ]:




