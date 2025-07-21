import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

##load the API KEYS
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

## loading the llm model
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

## creating the prompt template
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    Question:{input}
    """
)
## setting up vectors embedding
def create_vector_embedding():
    if "vectors" not in st.session_state:
        
        st.session_state.vectors=None
        # Step 1: Initiaze OpenAI Embeddings
        st.session_state.embeddings=OpenAIEmbeddings()

        #Step 2 Load PDF Documents from folder
        st.session_state.loader=PyPDFDirectoryLoader("research_paper") ## Data Ingestion Step
        st.session_state.docs=st.session_state.loader.load()

        #Step 3: Split Documents into chunks
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200)

        docs_to_split = st.session_state.docs[:50] if len(st.session_state.docs) >= 50 else st.session_state.docs
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(docs_to_split)

        # Step 4: Create FAISS vector store
        st.session_state.vectors=FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings)

st.title("RAG Document Q&A With Groq and Llama 3")

user_prompt=st.text_input("Enter your query from the research papers")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is Ready")


import time

if user_prompt:
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.error("Please create the vector database first by clicking the 'Document Embedding' button.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        print(f"Response time :{time.process_time() - start}")

        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("---------------------------")