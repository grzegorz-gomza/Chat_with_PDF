import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter


from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import (
    SelfHostedHuggingFaceEmbeddings,
    HuggingFaceEmbeddings,
)

import openai
from langchain_core import chat_history


from langgraph.checkpoint.memory import MemorySaver


import os

from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # read PDF File
        for page in pdf_reader.pages:  # Loop through all pages
            text += page.extract_text()  # append page into text variable
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    # For running with OpenAI API
    embedding_function = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')

    # For running local LLM with LM Studio
    # embedding_function = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs={"device": "cpu"},
    # )

    vectorstore = FAISS.from_texts(text_chunks, embedding=embedding_function)
    return vectorstore


def get_conversation_chain(vectorstore):
    ### LLM ###

    # For running with OpenAI API
    llm = ChatOpenAI()

    # For running local LLM with LM Studio
    # llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", api_key="lm-studio")

    ### Memory ###
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    memory = MemorySaver()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    # st.write(response)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    # load the variables from environment file
    load_dotenv()

    # set page config
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    # import css styling
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Set Header
    st.header("Chat with Multiple PDFs :books:")

    # Text bar for user input
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    # Sidebar for file upload
    with st.sidebar:
        # Define header
        st.subheader("Your documents")
        # File Uploader on the sidebar
        pdf_docs = st.file_uploader(
            "Choose a PDF file", type="pdf", accept_multiple_files=True
        )
        # Create the button to upload the file
        if st.button("Upload PDF"):
            with st.spinner("Uploading PDF..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                try:
                    # create vector store for embeddings
                    vectorstore = get_vectorstore(text_chunks)
                    st.success("PDF uploaded successfully", icon="✅")
                except Exception as e:
                    st.error("PDF upload failed", icon="❌")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )  # st.session_state to prevent var conversation to be lost in case streamlit will run the main code one more time, after clicking any button

        # Define header
        st.subheader("OpenAI API key")

        openai_api_key = st.text_input("Enter your OpenAI API key: ", type="password")
        if st.button("Save API key"):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.write("API key saved successfully!")


if __name__ == "__main__":
    main()
