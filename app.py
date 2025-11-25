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

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =====================================================
#                GOOGLE API KEY LOADER
# =====================================================
def get_google_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("Google API key not found in Secrets or .env")
        return api_key


# =====================================================
#                PDF + OCR EXTRACTION
# =====================================================
def extract_text_from_pdf_with_ocr(pdf_path):
    """
    1. Try normal text extraction
    2. If almost no text found → run OCR on the scanned pages
    """
    text = ""
    reader = PdfReader(pdf_path)
    page_texts = []

    # Try normal PDF text extraction
    for page in reader.pages:
        content = page.extract_text() or ""
        page_texts.append(content)

    if sum(len(t) for t in page_texts) > 50:  # Text PDF
        return "\n".join(page_texts)

    # Otherwise → OCR fallback
    ocr_text = ""
    images = convert_from_path(pdf_path, dpi=300)

    for img in images:
        ocr_text += pytesseract.image_to_string(img, lang="eng") + "\n"

    return ocr_text


# =====================================================
#                FILE TEXT EXTRACTOR
# =====================================================
def get_files_text(uploaded_files):
    text = ""

    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1].lower()

        if ext == ".pdf":
            # save uploaded PDF to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            text += extract_text_from_pdf_with_ocr(tmp_path)

            os.unlink(tmp_path)  # cleanup

        elif ext == ".docx":
            text += get_docx_text(file)

    return text


def get_docx_text(file):
    doc = docx.Document(file)
    return " ".join(para.text for para in doc.paragraphs if para.text.strip())


# =====================================================
#           TEXT CHUNKING & VECTOR STORE
# =====================================================
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(text_chunks, embeddings)


# =====================================================
#                   GEMINI RESPONSE
# =====================================================
def get_gemini_response(question, vectorstore):
    api_key = get_google_api_key()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-pro-latest",
        google_api_key=api_key,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the given context to answer.

If the answer is not found in the documents, reply:
"I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)


# =====================================================
#                       MAIN UI
# =====================================================
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
            "Upload PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        if st.button("🚀 Process Documents"):
            if not uploaded_files:
                st.error("❌ Please upload at least one file.")
                return

            try:
                get_google_api_key()
            except ValueError as e:
                st.error(str(e))
                st.stop()

            try:
                with st.spinner("🔧 Extracting & Processing..."):
                    raw_text = get_files_text(uploaded_files)

                    if not raw_text.strip():
                        st.warning("⚠️ No readable text found.")
                        return

                    chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(chunks)
                    st.session_state.vectorstore = vectorstore

                st.success("✅ Documents processed successfully!")

            except Exception as e:
                st.error(f"Error while processing: {str(e)}")
                st.code(traceback.format_exc())

    # Chat Area
    if st.session_state.vectorstore:
        user_question = st.chat_input("Ask something from your documents...")

        if user_question:
            with st.spinner("🤔 Thinking..."):
                response = get_gemini_response(
                    user_question,
                    st.session_state.vectorstore
                )

            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

        # Display conversation
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    else:
        st.info("👈 Upload & Process documents to start chatting!")


if __name__ == "__main__":
    main()
