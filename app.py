import streamlit as st
import time
import os

#import the backend
from agentic_backend import agent_app, optimized_retriever, bm25_retriever, vectorstore

#configuring page settings
st.set_page_config(
    page_title="AI Architecture Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

#custom css styling
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stChatFloatingInputContainer {
        background-color: transparent !important;
    }
    div.stButton > button:first-child {
        background-color: #4A90E2;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

#sidebar for system diagnostics
with st.sidebar:
    st.image("https://img.icons8.com/wired/64/000000/artificial-intelligence.png", width=60)
    st.title("System Health")
    st.subheader("Model Configuration")
    st.info("⚡ LLM: **Granite-4B (Ollama)**\n\n Embeddings: **Nomic-Embed**")
    
    st.subheader("RAG Retrievers Active")
    st.success(" **BM25 Lexical Index** (k=15)\n\n **Chroma Vector Store** (k=15)\n\n **Cross-Encoder Reranker** (top_n=5)")
    
    st.subheader("Safety Guardrails")
    st.warning(" **Python Loop Limits Enabled**\n- Max Local RAG Loops: 5\n- Max ArXiv Tool Calls: 3")

#main title header
st.title("Hardware & Edge AI Intelligence Agent")
st.caption("Enterprise-grade Agentic RAG combining Hybrid Keyword-Semantic Retrieval, Reranking, and ArXiv Academic Fallbacks.")

#initialize message history session state
if "messages" not in st.session_state:
    st.session_state.messages = []

#display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#helper function to stream string responses word by word
def stream_word_by_word(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04) #typing latency

#Input
if user_question := st.chat_input("Ask a technical hardware question ..."):
    
    #display user question
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    #live thought process update
    with st.status(" Agent Node Execution Pipeline Starting...", expanded=True) as status:
        
        st.write(" **[Node: analyze_query]** Stripping conversational filler & refining search intent...")
        time.sleep(0.5) #slight pause for readability
        
        try:
            #indicate state transition
            st.write(" **[Node: retrieve]** Executing parallel BM25 Keyword & Vector Semantic searches and Cross-Encoder reranking...")
            time.sleep(0.5)
            
            st.write(" **[Node: grade]** Validating context sufficiency...")
        except Exception as e:
            pass

        #initial graph memory state
        initial_state = {
            "original_question": user_question,
            "current_query": user_question,
            "retrieved_docs": [],
            "generation": "",
            "loop_count": 0,
            "research_messages": []
        }
        
        #activate Langgraph agent
        final_state = agent_app.invoke(initial_state)
        
        # read counter state for update
        final_loops = final_state.get("loop_count", 0)
        final_docs = final_state.get("retrieved_docs", [])
        
        #dynamically change status messages based on what route the agent autonomously picked
        if "live academic data" in final_state["generation"].lower() or final_loops >= 4:
            status.update(label="❌ Local Documentation Insufficient. Autonomous Web Fallback Triggered", state="error", expanded=True)
            st.write(f" **[Node: rewrite_query]** RAG database lacked precise records after several rewrites.")
            st.write(f" **[Node: research]** LLM called tool `search_literature` to fetch live academic papers from ArXiv.")
        else:
            status.update(label=f" Verification Successful! Retrieved {len(final_docs)} highly relevant local documentation chunks.", state="complete", expanded=False)

    #stream final response word by word with citations
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        raw_generation = final_state["generation"]
        
        #pipe generator to stream renderer
        st.write_stream(stream_word_by_word(raw_generation))
        
    # append final response to session state for conversation history
    st.session_state.messages.append({"role": "assistant", "content": raw_generation})