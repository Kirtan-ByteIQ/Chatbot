import sys
sys.path.append("document loaders")  
from pdf_loader import load_all_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
DOCUMENTS_FOLDER= "documents"
DB_FOLDER = "chroma_db"
print("S1: loading pdfs ...")
docs = load_all_pdfs(DOCUMENTS_FOLDER)
print(f"total chunks loaded from all pdfs: {len(docs)}")
print("\nS2: splitting into smaller chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
print(f"total chunks after splitting: {len(chunks)}")
print("\nS3: loading local embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nS4: creating embeddings + saving into chroma_db ...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_FOLDER
)

print("\nur local vector database is ready in the 'chroma_db' folder.")
