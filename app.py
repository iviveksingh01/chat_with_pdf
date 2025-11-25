import streamlit as st
st.set_page_config(page_title="Chat With Files", layout="wide")

import os
import traceback
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
import tempfile
import shutil  # for tesseract auto-detect

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load .env
load_dotenv()

# ------------------------------------------------------
#   TESSERACT AUTO-DETECTION (Linux/Windows/Mac Safe)
# ------------------------------------------------------
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print("⚠️ Warning: Tesseract not found in PATH.")


# ------------------------------------------------------
#   GOOGLE API KEY
# ------------------------------------------------------
def get_google_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("Google API key not found in Secrets or .env")
        return api_key


# ------------------------------------------------------
#   PDF + OCR
# ------------------------------------------------------
def extract_text_from_pdf_with_ocr(pdf_path):
    """
    1. Try PyPDF2 extraction.
    2. If text is too small, perform full OCR.
    """
    reader = PdfReader(pdf_path)
    page_texts = []

    for page in reader.pages:
        txt = page.extract_text() or ""
        page_texts.append(txt)

    # If searchable text exists
    if sum(len(t) for t in page_texts) > 50:
        return "\n".join(page_texts)

    # Otherwise OCR
    ocr_text = ""
    images = convert_from_path(pdf_path, dpi=300)

    for img in images:
        ocr_text += pytesseract.image_to_string(img, lang="eng") + "\n"

    return ocr_text


# ------------------------------------------------------
#   Extract docx + PDF
# ------------------------------------------------------
def get_files_text(uploaded_files):
    text = ""

    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1].lower()

        if ext == ".pdf":
            # Save temp PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            text += extract_text_from_pdf_with_ocr(tmp_path)

            os.unlink(tmp_path)

        elif ext == ".docx":
            text += get_docx_text(file)

    return text


def get_docx_text(file):
    doc = docx.Document(file)
    return " ".join(para.text for para in doc.paragraphs if para.text.strip())


# ------------------------------------------------------
#   Chunk + VectorStore
# ------------------------------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(text_chunks, embeddings)


# ------------------------------------------------------
#   Gemini RAG Response
# ------------------------------------------------------
def get_gemini_response(question, vectorstore):
    api_key = get_google_api_key()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-pro-latest",
        google_api_key=api_key,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context to answer.
If the answer isn't found, reply:
"I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)


# ------------------------------------------------------
#                  MAIN STREAMLIT APP
# ------------------------------------------------------
def main():
    st.header("💬 Chat with PDF / Scanned PDF / DOCX (Gemini RAG)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar
    with st.sidebar:
        st.subheader("📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        if st.button("🚀 Process Documents"):
            if not uploaded_files:
                st.error("Please upload at least one document.")
                return

            try:
                get_google_api_key()
            except ValueError as e:
                st.error(str(e))
                return

            try:
                with st.spinner("Extracting & Processing..."):
                    raw_text = get_files_text(uploaded_files)

                    if not raw_text.strip():
                        st.warning("⚠ No readable text found.")
                        return

                    chunks = get_text_chunks(raw_text)
                    st.session_state.vectorstore = get_vectorstore(chunks)

                st.success("✅ Documents processed successfully!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())

    # Chat Area
    if st.session_state.vectorstore:
        user_question = st.chat_input("Ask something from your documents...")

        if user_question:
            with st.spinner("🤔 Thinking..."):
                answer = get_gemini_response(
                    user_question,
                    st.session_state.vectorstore
                )

            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    else:
        st.info("👈 Please upload documents to begin.")


if __name__ == "__main__":
    main()
