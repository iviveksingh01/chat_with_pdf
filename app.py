import streamlit as st
st.set_page_config(page_title="Chat With Files", layout="wide")  # 👈 must be first Streamlit command

import os
import traceback
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv

# Stable LangChain 0.2+ imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (optional)
load_dotenv()

def main():
    try:
        st.header("💬 Chat with PDF/DOCX (Powered by Gemini)")

        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = None

        with st.sidebar:
            st.subheader("📁 Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF or DOCX files",
                type=["pdf", "docx"],
                accept_multiple_files=True
            )
            google_api_key = st.text_input(
                "Google AI Studio API Key",
                type="password",
                value=os.getenv("GOOGLE_API_KEY", ""),  # Auto-fill from .env if exists
                help="Get it from https://aistudio.google.com/app/apikey"
            )
            if st.button("🚀 Process Documents"):
                if not google_api_key.strip():
                    st.error("❌ Please enter your Google API key.")
                elif not uploaded_files:
                    st.error("❌ Please upload at least one file.")
                else:
                    try:
                        with st.spinner("🔧 Processing documents..."):
                            raw_text = get_files_text(uploaded_files)
                            if not raw_text.strip():
                                st.warning("⚠️ No text found in the uploaded files.")
                                st.session_state.vectorstore = None
                                return
                            text_chunks = get_text_chunks(raw_text)
                            if not text_chunks:
                                st.error("❌ Failed to split text into chunks.")
                                return
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.vectorstore = vectorstore
                        st.success("✅ Documents processed successfully!")
                    except Exception as e:
                        st.error(f"💥 Error during processing: {str(e)}")
                        st.code(traceback.format_exc())

        # Chat interface
        if st.session_state.vectorstore is not None:
            user_question = st.chat_input("Ask a question about your documents...")
            if user_question:
                try:
                    with st.spinner("🤔 Thinking..."):
                        response = get_gemini_response(
                            user_question,
                            st.session_state.vectorstore,
                            google_api_key.strip()
                        )
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"💥 Error generating response: {str(e)}")
                    st.code(traceback.format_exc())

            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        else:
            st.info("👈 Upload and process your documents to start chatting!")

    except Exception as e:
        st.error("🚨 An unexpected error occurred in the main function:")
        st.code(traceback.format_exc())

# === Helper Functions ===

def get_files_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            text += get_pdf_text(file)
        elif ext == ".docx":
            text += get_docx_text(file)
    return text

def get_pdf_text(pdf):
    reader = PdfReader(pdf)
    return "".join(page.extract_text() or "" for page in reader.pages)

def get_docx_text(file):
    doc = docx.Document(file)
    return " ".join(para.text for para in doc.paragraphs if para.text.strip())

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

def get_gemini_response(question, vectorstore, api_key):
    if not api_key:
        raise ValueError("Google API key is missing.")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-pro-latest",
        google_api_key=api_key,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use ONLY the following context to answer the question.
        If the answer is not in the context, say: "I don't know based on the provided documents."

        Context:
        {context}

        Question: {question}
        Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

# --- Run App ---
if __name__ == '__main__':
    main()
