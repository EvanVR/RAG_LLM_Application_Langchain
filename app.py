from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import streamlit as st
import os

load_dotenv()

# Function to create a vector store from a website URL
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)  
    document = loader.load()  
    
    text_splitter = RecursiveCharacterTextSplitter()  
    document_chunks = text_splitter.split_documents(document)  # Split the document into chunks
    
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())  # Create a Chroma vector store using OpenAI embeddings
    
    return vector_store

# Function to create a vector store from a PDF file
def get_vectorstore_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)  
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()  # Extract text from each page
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )  # Initialize a text splitter
    chunks = text_splitter.split_text(text)  # Split the text into chunks
    
    embeddings = OpenAIEmbeddings()  # Initialize OpenAI embeddings
    knowledge_base = FAISS.from_texts(chunks, embeddings)  # Create a FAISS vector store from the chunks
    
    return knowledge_base

# Function to create a vector store from a YouTube video URL
def get_vectorstore_from_youtube(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)  
    docs = loader.load()  # Load the document
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, separator=" ", chunk_overlap=50)  # Initialize a text splitter
    split_docs = text_splitter.split_documents(docs)  # Split the document into chunks
    
    embeddings = OpenAIEmbeddings()  # Initialize OpenAI embeddings
    vector_store = Chroma.from_documents(split_docs, embeddings)  # Create a Chroma vector store from the chunks
    
    return vector_store

# Function to create a context-aware retriever chain
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()  # Initialize the language model
    
    retriever = vector_store.as_retriever()  # Create a retriever from the vector store
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])  # Create a prompt template for generating a search query
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)  # Create a history-aware retriever chain
    
    return retriever_chain

# Function to create a conversational Retrieval-Augmented Generation (RAG) chain
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()  # Initialize the language model
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])  # Create a prompt template for generating answers
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)  # Create a chain for generating answers from the context
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)  # Create a retrieval chain combining the retriever and answer generation chains

# Function to get a response for a user query
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)  # Get the retriever chain
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)  # Get the conversational RAG chain
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })  # Get the response from the RAG chain
    
    return response['answer']

st.set_page_config(page_title="Chat with Websites, PDFs, and Videos", page_icon="🤖")
st.title("IntelliBlend: \n Chat with Websites, PDFs, and Videos")

with st.sidebar:
    st.sidebar.title("Evan Velagaleti \n ev379@drexel.edu")
    website_url = st.text_input("Website URL")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    video_url = st.text_input("YouTube Video URL")

if website_url:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)  # Create vector store from the URL

    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)  
        st.session_state.chat_history.append(HumanMessage(content=user_query))  # Add user query to chat history
        st.session_state.chat_history.append(AIMessage(content=response))  

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

elif pdf:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_pdf(pdf)  # Create vector store from the PDF

    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)  # Get response for the user query
        st.session_state.chat_history.append(HumanMessage(content=user_query))  
        st.session_state.chat_history.append(AIMessage(content=response))  

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

elif video_url:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_youtube(video_url)  # Create vector store from the YouTube URL

    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query) 
        st.session_state.chat_history.append(HumanMessage(content=user_query))  
        st.session_state.chat_history.append(AIMessage(content=response))  

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

else:
    st.info("Please enter a website URL, upload a PDF, or enter a YouTube video URL.")
