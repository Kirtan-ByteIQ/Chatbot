import streamlit as st

st.title("My Project")


import os
import sys

sys.path.append("document loaders")  
from pdf_loader import load_all_pdfs

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

DOCUMENTS_FOLDER = "documents"
DB_FOLDER = "chroma_db"

st.set_page_config(page_title="Local Document Chatbot", page_icon="📄")
st.title("📄 Local  Chatbot")


# ---------------- cached, so these only load ONCE per session ----------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2")


def get_vectorstore():
    embedding_model = get_embedding_model()
    return Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_model)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.
The context may come from normal page text OR from a description/OCR text
of an image, chart or graph found inside a document.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)


# ---------------- sidebar: upload pdfs + (re)build the database ----------------
with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join(DOCUMENTS_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file/files")

    st.divider()

    if st.button("🔄 scan doc", use_container_width=True):
        with st.spinner("Working..."):
            docs = load_all_pdfs(DOCUMENTS_FOLDER)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            embedding_model = get_embedding_model()
            Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=DB_FOLDER,
            )
        st.success(f"Database built from {len(chunks)} chunks.")



# ---------------- main chat window ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not os.path.exists(DB_FOLDER):
        answer = "No database found yet. Upload PDFs and click 'Build / Rebuild database' in the sidebar first."
    else:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )

        with st.spinner("Thinking..."):
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke({"context": context, "question": query})
            response = get_llm().invoke(final_prompt)
            answer = response.content

            with st.expander("Sources used"):
                for d in docs:
                    st.write(d.metadata)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
