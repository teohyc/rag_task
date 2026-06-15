from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import requests
import arxiv

#for graph visualization
from PIL import Image
from io import BytesIO

#configuration
PERSIST_DIR = "vectordb"
COLLECTION_NAME = "edge_ai_knowledge"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "granite4:latest"  

#initialize all needed components
print("Initializing Agentic Hardware Pipeline...")

#initialize embedding model and llm
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0.1) #low temp for more factual and less creative responses due to technical nature of the question

print("Building Hybrid Retriever...")
#initialize vector database connection
vectorstore = Chroma(persist_directory=PERSIST_DIR, collection_name=COLLECTION_NAME, embedding_function=embeddings)

#initialize BM25 retriever from ChromaDB
db_data = vectorstore.get()
all_documents = []
    
# Reconstruct the Document objects
for i in range(len(db_data['ids'])):
    doc = Document(
        page_content=db_data['documents'][i], 
        metadata=db_data['metadatas'][i]
    )
    all_documents.append(doc)
        
bm25_retriever = BM25Retriever.from_documents(all_documents)
bm25_retriever.k = 15 #top 15 results from BM25

#cross-encoder retriever setup
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

#build hybrid retriever
hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5] #give equal weight to keywords and semantics
    )

#cross-encoder reranker setup
reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
optimized_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid_retriever) #15 chunks down to 5 most relevant ones

#literature search tool (only used when RAG retrieval fails)
@tool
def search_literature(query: str) -> str:
    """Search hardware, Edge AI, and LLM compression academic literature."""
    print(f"   [Executing Tool]: Searching ArXiv for -> '{query}'")

    try:
        #arxiv search query
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        #using arxiv client
        client = arxiv.Client()
        results = list(client.results(search))

        if not results:
            return "No relevant papers found for this query. Try different keywords."

        summaries = []
        for result in results:
            #clean response and remove newlines for better formatting in the LLM context
            abstract = result.summary.replace('\n', ' ')
            
            #saving context window
            if abstract and len(abstract) > 1300:
                abstract = abstract[:1300] + "..."
                
            summaries.append(
                f"[Source: {result.title} ({result.published.year})]\n"
                f"CONTENT: {abstract}\nURL: {result.pdf_url}\n---"
            )

        final_tool_output = "\n".join(summaries)

        #diagnostic print
        print("\n" + "="*40)
        print("[DIAGNOSTIC: RAW ARXIV RESULTS FETCHED]")
        print("="*40)
        print(final_tool_output)
        print("="*40 + "\n")

        return final_tool_output

    except Exception as e:
        return f"Literature search error: {e}"

#define the AgentState (memory)
class AgentState(TypedDict):
    original_question: str
    current_query: str
    retrieved_docs: list
    generation: str
    loop_count: int
    research_messages: list

#NODES OF THE GRAPH

def analyze_query_node(state: AgentState) -> AgentState:
    """Intercepts the raw user question and optimizes it for Hybrid Search."""
    print("\n--- [AGENT: ANALYZING USER INTENT & OPTIMIZING QUERY] ---")
    
    prompt = SystemMessage(content=f"""You are an expert database search query generator.
    Your task is to convert the user's conversational question into a highly optimized search query for a Vector and BM25 database.
    
    RULES:
    1. Strip away all conversational filler (e.g., "What is", "Can you explain", "tell me").
    2. Keep ONLY the core technical concepts, hardware names, and algorithmic terms.
    3. Return ONLY the optimized search string. Do not include quotes, preambles, or explanations.
    
    User Question: {state['original_question']}
    """)
    
    response = llm.invoke([prompt])
    
    #clean up the output in case of any formatting issues or extra symbols
    optimized_query = response.content.strip().replace('"', '').replace('\n', ' ')
    
    print(f"Optimized Query: '{optimized_query}'")
    
    return {"current_query": optimized_query}

def retrieve_node(state: AgentState) -> AgentState:
    """Fetches documents using the optimized Cross-Encoder pipeline."""
    print(f"\n--- [AGENT: RETRIEVING] Query: '{state['current_query']}' ---")
    
    docs = optimized_retriever.invoke(state["current_query"])

    #for debugging and transparency
    print("--- [DIAGNOSTIC] CHOSEN SOURCES ---")
    for i, doc in enumerate(docs):
        src = doc.metadata.get('source', 'Unknown').split("\\")[-1].split("/")[-1] #works for window, linux and mac
        pg = doc.metadata.get('page', 'N/A')
        print(f"Chunk {i+1}: {src} (Page {pg})")
    print("-----------------------------------\n")
    
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown File').split("\\")[-1].split("/")[-1]
        page = doc.metadata.get('page', 'N/A')
        formatted_docs.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}\n")

    print(f"Successfully retrieved {len(docs)} chunks.\n")
        
    return {"retrieved_docs": formatted_docs}

def grade_documents_node(state: AgentState) -> AgentState:
    print("--- [AGENT: GRADING DOCUMENTS] ---")
    
    prompt = SystemMessage(content=f"""You are a strict grading evaluator. 
    Does the following context explicitly contain the facts needed to answer the user's question?
    You must grade by prioritizing the question's intent.
    Every response MUST end with either <YES> if the context is sufficient to answer the question, or <NO> if it is not.
    
    Question: {state['original_question']}
    Context: {state['retrieved_docs']}
    
    INSTRUCTIONS:
    1. Write a brief 1-sentence explanation of your reasoning.
    2. MUST end your response with the XML tag <YES> if sufficient, or <NO> if insufficient.
    
    Example Output:
    The context discusses memory bandwidth but does not mention the RTX 3050.
    <NO>
    """)
    
    response = llm.invoke([prompt])
    
    #sanitizing the output
    grade_output = response.content.strip().lower().replace("[", "<").replace("]", ">").replace("’", "'").replace("‘", "'")
    
    print(f"Agent Reasoning:\n{response.content.strip()}")
    
    #safety check for hesitant language
    hesitant_phrases = [
        "doesn't explicitly ", 
        "doesn't directly ", 
        "does not explicitly ", 
        "does not directly ", 
        "does not provide",
        "doesn't contain",
        "does not contain",
        "however,",
        "but it "
    ]
    
    if any(phrase in grade_output for phrase in hesitant_phrases):
        print("PYTHON OVERRIDE: Hesitant phrase detected, forcing NO")
        return {"current_query": "insufficient"}

    #greedy parsing for xml tags
    if "<yes>" in grade_output:
        print(">> Grade Evaluation: SUFFICIENT")
        return {"current_query": "sufficient"}
    else:
        print(">> Grade Evaluation: INSUFFICIENT")
        return {"current_query": "insufficient"}
    
def rewrite_query_node(state: AgentState) -> AgentState:
    """If documents failed, the agent rewrites the query to be more specific."""
    print("--- [AGENT: REWRITING QUERY] ---")
    
    prompt = SystemMessage(content=f"""You are an expert search query generator.
    The previous search for the question below failed to find good technical context.
    Rewrite the question into a better, more specific keyword search query.
    Return ONLY the new search query text.
    
    Original Question: {state['original_question']}
    Question Asked to LLM That Failed: {state['current_query']}
    """)
    
    response = llm.invoke([prompt])
    new_query = response.content.strip()
    print(f"New Search Query: {new_query}")
    
    return {"current_query": new_query, "loop_count": state["loop_count"] + 1} #increase loop count

def external_research_node(state: AgentState) -> dict:
    print(f"--- [AGENT: EXTERNAL RESEARCH] Iteration: {state['loop_count']} ---")
    
    #define the query for the tool based on the original question and previous failed query
    existing_msg = state.get("research_messages", [])

    #diagnostic to check if the LLM is reading tool output correctly in the loop
    if existing_msg and existing_msg[-1].type == "tool":
        print(f"\n[DIAGNOSTIC: LLM IS READING TOOL OUTPUT]\n{existing_msg[-1].content}\n")

    #bind the tool to the LLM
    llm_with_tools = llm.bind_tools([search_literature])
    
    system_msg = SystemMessage(content=f"""
    You are an AI hardware researcher. 
    CRITICAL INSTRUCTION: The secure local database just FAILED to contain the answer for: "{state['original_question']}"
    
    Your tool is 'search_literature'. Use it to search the web for live academic papers.
    
    When you have gathered enough information, synthesize a final technical answer.
    YOUR FINAL ANSWER MUST FOLLOW THIS STRUCTURE:
    1. You MUST start by explicitly stating: "I could not find this information in the local documentation. However, I have retrieved the following live academic data:"
    2. Answer the question using the tool data.
    3. Cite your web sources using the URLs provided.
    
    If you cannot find the answer after searching, you MUSTadmit defeat gracefully and explicitly.
    """)
    
    #if this is the first time entering the node, start the conversation
    if not existing_msg:
        response = llm_with_tools.invoke([system_msg, HumanMessage(content=state["original_question"])])
        return {"research_messages": [response], "loop_count": state["loop_count"] + 1} #increase loop count
    else:
        #continue the conversation with tool results
        response = llm_with_tools.invoke([system_msg] + existing_msg)
        return {"research_messages": existing_msg + [response], "loop_count": state["loop_count"] + 1} #increase loop count

def finalize_research_node(state: AgentState) -> dict:
    """Extracts the final answer from the research loop."""
    print("--- [AGENT: RESEARCH COMPLETE. FINALIZING] ---")
    last_msg = state["research_messages"][-1]
    return {"generation": last_msg.content}

def generate_node(state: AgentState) -> AgentState:
    """Generates the final response with citations."""
    print("--- [AGENT: GENERATING FINAL RESPONSE] ---")
    
  
    prompt = SystemMessage(content=f"""You are an expert Edge AI and Hardware engineer.
    Answer the user's question comprehensively using ONLY the provided context. 
    Write in a confident and technical conversational style in point form.
    Prioritze answering the user's question.
    
    Context:
    {state['retrieved_docs']}
    
    Question: {state['original_question']}

    Note: Every factual claim MUST be followed by its inline citation: [Source: filename.pdf, Page: X] 
    Read the question carefully and pay attention to every word and numbers in the question.
    """)
        
    
    response = llm.invoke([prompt])
    return {"generation": response.content}


#branch function to decide if there is sufficient context
#RAG internal though branch
def decide_next_step(state: AgentState) -> str:
    """Routes the graph based on the Grader's decision."""
    if state["current_query"] == "sufficient":
        return "generate"
    
    # After 5 failed local rewrites, give up on local and go to the web tool
    if state["loop_count"] >= 4: #maximum 5 local queries
        print("--- [AGENT: LOCAL MAX LOOPS REACHED. ROUTING TO WEB RESEARCH] ---")
        return "research"
        
    return "rewrite"

#research branch
def route_research(state: AgentState) -> str:
    """Decides whether to execute the tool, finish, or abort based on the limit."""
    last_msg = state["research_messages"][-1]
    
    # If the LLM outputted a tool call, route to the tool execution node
    if getattr(last_msg, "tool_calls", None):
        if state["loop_count"] >= 8: #maximum 3 web search
            print("MAX ONLINE SEARCHES REACHED. FORCING FINAL ANSWER.")
            return "generate_fallback"
        return "execute_tool"
        
    #if the LLM didn't call a tool, it generated a standard text answer
    return "finalize_research"

#BUILDING THE GRAPH
workflow = StateGraph(AgentState)

#tool
tool_node = ToolNode([search_literature], messages_key="research_messages") #define location  of tool calls in the state

#Nodes
workflow.add_node("analyze_query", analyze_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)     
workflow.add_node("generate", generate_node)
workflow.add_node("research", external_research_node)
workflow.add_node("execute_tool", tool_node)
workflow.add_node("finalize", finalize_research_node)

#entry point
workflow.set_entry_point("analyze_query")

#edge to analyse quer before retieval
workflow.add_edge("analyze_query", "retrieve")

#local RAG loop
workflow.add_edge("retrieve", "grade")

#edges from the grader node based on its decision
workflow.add_conditional_edges(
    "grade", 
    decide_next_step, 
    {
        "generate": "generate", 
        "rewrite": "rewrite",
        "research": "research" #loop
    }
)

#retrieval after rewrite
workflow.add_edge("rewrite", "retrieve")

#web search react loop
workflow.add_conditional_edges(
    "research",
    route_research,
    {
        "execute_tool": "execute_tool",
        "finalize_research": "finalize",
        "generate_fallback": "finalize"
    }
)

#activate if llm decides to execute a tool call in the research node
workflow.add_edge("execute_tool", "research")

#end points
workflow.add_edge("generate", END)
workflow.add_edge("finalize", END)

agent_app = workflow.compile()

'''
#generate graph visualization
png = agent_app.get_graph().draw_mermaid_png()
img = Image.open(BytesIO(png))
img.show()
img.save("rag_agent_diagram.png")

'''
#test run
if __name__ == "__main__":
    while True:
        test_question = input("Enter your question for the RAG Agent:\n")
    
        initial_state = {
            "original_question": test_question,
            "current_query": test_question,
            "retrieved_docs": [],
            "generation": "",
            "loop_count": 0
        }
    
        print("\nStarting Agentic Run...\n" + "="*50)
        final_state = agent_app.invoke(initial_state)
        print("\n" + "="*50)
        print("\nFINAL OUTPUT:\n")
        print(final_state["generation"])
