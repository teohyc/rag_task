import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

#configurations
DOCS_PATH = "rag_docs"
PERSIST_DIR = "vectordb"
COLLECTION_NAME = "edge_ai_knowledge"
EMBEDDING_MODEL = "nomic-embed-text:latest"

def build_vector_db():
    print(f"--- Starting Ingestion Pipeline ---")
    start_time = time.time()
    documents = []

    #load documents
    if not os.path.exists(DOCS_PATH):
        print(f"Error: Directory '{DOCS_PATH}' not found. Please create it and add PDFs.")
        return

    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDFs found in '{DOCS_PATH}'.")
        return

    for file in pdf_files:
        file_path = os.path.join(DOCS_PATH, file)
        print(f"Loading and parsing: {file}")

        #PyPDFLoader automatically extracts the filename and page numbers into the metadata for citation
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    print(f"\nSuccessfully loaded {len(documents)} total pages.")

    #optimized chunking for long context windows and better retrieval relevance for research paper length
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} optimized chunks.")

    #diplaying sample metadata to verify citation readiness
    if chunks:
        print("\nVerifying Metadata ")
        print(f"Sample Chunk Metadata: {chunks[0].metadata}")

    #generate emebeddings for chroma db
    print(f"\nInitializing embedding model: {EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    print("Building and persisting Chroma vector database...")

    #create vector db
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    end_time = time.time()
    print(f"\nVector Database successfully built in {round(end_time - start_time, 2)} seconds")
    print(f"Database persisted at: ./{PERSIST_DIR}/")

if __name__ == "__main__":
    build_vector_db()