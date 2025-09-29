import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


# ------------------- Step1: Save Uploaded PDF -------------------
pdfs_directory = "pdfs"

def saved_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    file_path = os.path.join(pdfs_directory, file.name)

    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    return file_path


# ------------------- Step2: Load PDF -------------------
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# ------------------- Step3: Create Chunks -------------------
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


# ------------------- Step4: Setup Embeddings Model -------------------
ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model(model_name):
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings


# ------------------- Step5: Index Documents in FAISS -------------------
FAISS_DB_PATH = "vectorstore/db_faiss"

def create_vector_store(db_path, text_chunks, model_name):
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(model_name))
    faiss_db.save_local(db_path)
    return faiss_db


# ------------------- Step6: Retrieve Docs -------------------
def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


# ------------------- Step7: Custom Prompt Template -------------------
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer.
Don’t provide anything outside of the given context.

Question: {question}
Context: {context}
Answer:
"""


# ------------------- Step8: Setup LLM -------------------
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")


# ------------------- Step9: Answer Query -------------------
def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})


# ------------------- Step10: Streamlit UI -------------------
st.title("📑 DeepSeek-R1 reasoning - PDF Q/A Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf",
    accept_multiple_files=False
)

user_query = st.text_input("💬 Enter your question:", placeholder="Ask anything about the PDF...")
ask_question = st.button("Ask reasoning AI")


# ------------------- Step11: RAG Pipeline -------------------
if ask_question:
    if uploaded_file and user_query:
        # Save uploaded file
        file_path = saved_pdf(uploaded_file)

        # Process PDF
        documents = load_pdf(file_path)
        text_chunks = create_chunks(documents)
        faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks, ollama_model_name)

        # Retrieve + Answer
        retrieved_docs = retrieve_docs(faiss_db, user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        # Display Chat
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(response)

    else:
        st.error("⚠️ Please upload a valid PDF file and enter your question.")
