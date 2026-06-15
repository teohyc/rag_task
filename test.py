from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
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
    
    #connect to ChromaDB
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    #building BM25 retriever from the raw documents stored in ChromaDB
    print("Building BM25 Lexical Retriever from ChromaDB chunks...")
    db_data = vectorstore.get()
    all_documents = []
    
    #reconstruct the document objects
    for i in range(len(db_data['ids'])):
        doc = Document(
            page_content=db_data['documents'][i], 
            metadata=db_data['metadatas'][i]
        )
        all_documents.append(doc)
        
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = 15 #top 15 results from BM25
    
    #vector search
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    #building hybrid retriever that combines BM25 and vector search results
    print("Fusing Vector and BM25 Retrievers...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5] #give equal weight to keywords and semantics
    )

    #reranker model
    print("Loading Cross-Encoder Reranker...")
    reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    #compressing 15 chunks into 3 most relevent ones
    compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

    #combined into optimized cross-encoder retriever
    optimized_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )
    
    print("\nRetrieving and Re-ranking context...")
    

    #initialize llm
    print(f"Initializing LLM: {LLM_MODEL}...")
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.2 #slight creativity
    )
    
    #define query
    query = input("Query: ")
    print(f"\n[User Query]: {query}")
    
    #FULL QUERY DIAGNOSTIC
    
    #bm25 diagnostic
    print("\n--- [DIAGNOSTIC] BM25 LEXICAL SEARCH (TOP 15) ---")
    bm25_raw = bm25_retriever.invoke(query)
    for i, doc in enumerate(bm25_raw):
        src = doc.metadata.get('source', 'Unknown').split("\\")[-1].split("/")[-1]
        pg = doc.metadata.get('page', 'N/A')
        print(f"BM25 Rank {i+1}: {src} (Page {pg})")
        
    #vector search diagnostic
    print("\n--- [DIAGNOSTIC] VECTOR SEMANTIC SEARCH (TOP 15) ---")
    vector_raw = vectorstore.similarity_search_with_score(query, k=15)
    for i, (doc, score) in enumerate(vector_raw):
        src = doc.metadata.get('source', 'Unknown').split("\\")[-1].split("/")[-1]
        pg = doc.metadata.get('page', 'N/A')
        # Note: Depending on Chroma's metric (L2 vs Cosine), lower distance is usually better
        print(f"Vector Rank {i+1}: {src} (Page {pg}) | Distance: {score:.4f}")

    #hybrid retriever diagnostic
    print("\nRetrieving context from Hybrid Retriever + Cross-Encoder...")
    
    #fetch the chunks and extract metadata
    docs = optimized_retriever.invoke(query)
    
    if not docs:
        print("Error: No documents found! The database might be empty.")
        return
    
    #cross encoder selection diagnostic
    print("\n--- [DIAGNOSTIC] FINAL CHOSEN SOURCES (TOP 5) ---")
    for i, doc in enumerate(docs):
        src = doc.metadata.get('source', 'Unknown').split("\\")[-1].split("/")[-1] 
        pg = doc.metadata.get('page', 'N/A')
        relevance = doc.metadata.get('relevance_score', 'N/A')
        
        if isinstance(relevance, float):
            print(f"Final {i+1}: {src} (Page {pg}) ")
        else:
            print(f"Final {i+1}: {src} (Page {pg}) ")
    print("-----------------------------------\n")

    context = ""
    for i, doc in enumerate(docs):
        #extracting the metadata
        source = doc.metadata.get('source', 'Unknown File').split("\\")[-1].split("/")[-1]
        page = doc.metadata.get('page', 'N/A')
        context += f"\n--- [Source: {source}, Page: {page}] ---\n{doc.page_content}\n"
        
    print(f"Successfully retrieved {len(docs)} highly optimized chunks. Handing off to {LLM_MODEL}...\n")
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
    
    #stream the LLM response token by token
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        print(chunk.content, end="", flush=True)
        
    end_time = time.time()
    print("\n" + "-" * 70)
    print(f"Full RAG cycle completed in {round(end_time - start_time, 2)} seconds.")

if __name__ == "__main__":
    test_rag_pipeline()