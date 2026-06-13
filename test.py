from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import time

#configuration
PERSIST_DIR = "vectordb"
COLLECTION_NAME = "edge_ai_knowledge"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:latest"  

def test_rag_pipeline():
    print(f"--- Booting up Integration Test ---")
    start_time = time.time()
    
    #initialize embedding model and connect to vector database
    print(f"Connecting to vector database using {EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    #build cross-encoder retriever
    #vector search
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    #reranker model
    print("Loading Cross-Encoder Reranker...")
    reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    #compressing 15 chunks into 3 most relevent ones
    compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

    #combined into optimized cross-encoder retriever
    optimized_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    print("\nRetrieving and Re-ranking context...")
    

    #initialize llm
    print(f"Initializing LLM: {LLM_MODEL}...")
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.2 #slight creativity
    )
    
    #define query
    query = "According to the documents, what is the main difference between LoRA and QLoRA?"
    print(f"\n[User Query]: {query}")
    print("\nRetrieving context from ChromaDB...")
    
    # fetch the chunks and extract metadata
    docs = optimized_retriever.invoke(query)
    
    if not docs:
        print("Error: No documents found! The database might be empty.")
        return
    
    #print retrieved docs
    print("--- [DIAGNOSTIC] CHOSEN SOURCES ---")
    for i, doc in enumerate(docs):
        src = doc.metadata.get('source', 'Unknown').split("\\")[-1].split("/")[-1] #works for window, linux and mac
        pg = doc.metadata.get('page', 'N/A')
        print(f"Chunk {i+1}: {src} (Page {pg})")
    print("-----------------------------------\n")

    context = ""
    for i, doc in enumerate(docs):
        #extracting the metadata we built during the ingestion phase
        source = doc.metadata.get('source', 'Unknown File').split("\\")[-1].split("/")[-1]
        page = doc.metadata.get('page', 'N/A')
        context += f"\n--- [Source: {source}, Page: {page}] ---\n{doc.page_content}\n"
        
    print(f"Successfully retrieved {len(docs)} chunks. Handing off to Gemma 3...\n")
    print("-" * 70)
    
    #strict prompt forcing citations
    prompt = f"""You are an expert AI storage and optimization engineer. 
    Analyze the provided context and provide a comprehensive, detailed comparison answering the user's question.
    
    Guidelines:
    - Provide a structured breakdown comparing the mechanisms (e.g., base data types, memory footprints, or quantization techniques).
    - Every factual claim or metric you bring up MUST be followed by its inline citation, for example: [Source: filename.pdf, Page: X].
    - Maintain a professional, technical tone.
    - Rely strictly on the facts provided below. Do not extrapolate beyond the context.
    - Provide the citations in the format at the end of each claim: [Source: filename.pdf, Page: X] immediately after the claim they support.
    
    Context:
    {context}
    
    Question: {query}
    """
    
    #stream the LLM response token-by-token
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        print(chunk.content, end="", flush=True)
        
    end_time = time.time()
    print("\n" + "-" * 70)
    print(f"Full RAG cycle completed in {round(end_time - start_time, 2)} seconds.")

if __name__ == "__main__":
    test_rag_pipeline()