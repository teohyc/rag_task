from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import time

from agentic_backend import agent_app

#initialize application
app = FastAPI(
    title="Agentic API",
    description="REST endpoint for the Hardware Intelligence RAG Agent",
    version="1.0.0"
)

#define the expected JSON input schema
class ChatRequest(BaseModel):
    question: str

#define the generator function that streams the text
async def stream_generator(text: str):
    """Yields words one by one to simulate LLM token streaming over HTTP."""
    words = text.split(" ")
    for word in words:
        #yield the word formatted as Server-Sent Events (SSE)
        yield f"data: {word} \n\n"
        await asyncio.sleep(0.04) #simulate typing latency

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Receives a question, runs the LangGraph Agentic RAG, 
    and streams the final response back to the client.
    """
    print(f"\n[API LOG] Received Question: {request.question}")
    
    #initial graph memory state
    initial_state = {
        "original_question": request.question,
        "current_query": request.question,
        "retrieved_docs": [],
        "generation": "",
        "loop_count": 0,
        "research_messages": []
    }
    
    #execute the LangGraph state machine synchronously
    start_time = time.time()
    final_state = agent_app.invoke(initial_state)
    end_time = time.time()
    
    print(f"[API LOG] Agent execution completed in {round(end_time - start_time, 2)} seconds.")
    
    raw_generation = final_state.get("generation", "Error: No generation produced.")
    
    #return the StreamingResponse piping generator to client
    return StreamingResponse(
        stream_generator(raw_generation), 
        media_type="text/event-stream"
    )

#health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "online", "agent": "Granite-4B Hybrid RAG"}