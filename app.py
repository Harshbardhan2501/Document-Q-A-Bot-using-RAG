from dotenv import load_dotenv

load_dotenv()

import langchain_community
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.exceptions import ModelAPIError
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from time import sleep
import os

try:
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
except StreamlitSecretNotFoundError:
    google_api_key = None

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", retries=1)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def document_process(path):

    ## documrnt Loadig

    loader = PyPDFLoader(path)
    docs = loader.load()

    ## splitting

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)


    ##embidding and vector store

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True



st.subheader("Document Q&A Chatbot - Ask Anything")
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False


### Document uploading
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select Your PDF File", type="pdf")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing the document..."):
            document_process("./uploaded_document.pdf")

        st.markdown("Document processed successfully...")
        sleep(2)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db:
    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]

        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask Anything ...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        try:
            with st.spinner("Searching the PDF and waiting for Gemini..."):
                documents = st.session_state.vector_db.similarity_search(
                    query=query,
                    k=2,
                )
                context = ""

                for doc in documents:
                    context += doc.page_content + "\n\n"

                prompt = f"""you are a helpful assistant and you provide answers for user question based on the provided context. context: {context}and question: {query}"""
                result = llm.invoke(prompt)

            answer = result.text
        except ModelAPIError:
            st.error(
                "Gemini is temporarily unavailable or busy. Please try asking again shortly."
            )
        else:
            st.session_state.messages.append({"role": "ai", "content": answer})
            st.chat_message("ai").markdown(answer)



### chat UI