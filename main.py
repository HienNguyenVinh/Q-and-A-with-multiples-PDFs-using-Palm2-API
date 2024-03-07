import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings, GooglePalmEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GooglePalm
from langchain.prompts.prompt import PromptTemplate
from html_templates import user_template, bot_template, css
from dotenv import load_dotenv
import os

load_dotenv()

palm_api_key = os.environ['GOOGLE_API_KEY']


def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(raw_text):
    text_spliter = RecursiveCharacterTextSplitter(
                                         chunk_size=1000,
                                         chunk_overlap=20)
    chunks = text_spliter.split_text(raw_text)

    return chunks

def get_vector_db(chunks):
    # embedding = HuggingFaceInstructEmbeddings()
    embedding = GooglePalmEmbeddings()
    vectordb = FAISS.from_texts(texts=chunks, embedding=embedding)

    return vectordb

def get_conversation_chain(vectordb):
    prompt_template = """
        You are a helpful AI assistant named Mai. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.

        context: {context}
        question: {question}
        """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = GooglePalm()
    retriever = vectordb.as_retriever()

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True,)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                                retriever=retriever,
                                                                memory=memory,
                                                               combine_docs_chain_kwargs={'prompt': PROMPT}
                                                               )
    return conversation_chain


def main():
    st.set_page_config(page_title="Q&A with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Q&A with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents: ")

    if user_question:
        response = st.session_state.conversation({'question': user_question})
        # st.write(response)
        st.session_state.chat_history = response['chat_history']

        for i in reversed(range(len(st.session_state.chat_history))):
            message = st.session_state.chat_history[i]

            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.write(user_template.replace("{{MSG}}", "Hello Mai"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process", accept_multiple_files=True)
        btn_process = st.button("Process")

        if btn_process:
            with st.spinner("Processing"):
                # Lấy text từ file pdf
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # Tạo các đoạn văn bản ngắn (chunks) raw_text
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                #Tạo vector database của text
                vectordb = get_vector_db(text_chunks)

                # Tạo chain
                st.session_state.conversation = get_conversation_chain(vectordb)
            st.success("Done!")



if __name__ == "__main__":
    main()