# main.py

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from agent.graph import graph
from langchain_core.messages import HumanMessage, AIMessage
from collections import defaultdict
import asyncio
from typing import Dict, List, Any
import json

app = FastAPI()

# ✅ In-memory conversation store for each user
conversations: Dict[str, List[HumanMessage | AIMessage]] = defaultdict(list)

@app.post("/chat")
async def chat(request: Request):
    """
    Handles standard chat requests without streaming.
    Processes user input, invokes the LangGraph, and returns the full AI reply.
    """
    print("\n--- Received /chat request ---")
    body = await request.json()
    user_id = body.get("user_id", "default")
    user_input = body.get("message")
    
    if not user_input:
        print("Chat: Missing 'message' in request body.")
        return {"error": "Missing 'message' in request body."}
    
    # Store the user's message in the conversation history
    # The parser node will handle the JSON parsing internally
    input_message_content = json.dumps(user_input) if isinstance(user_input, dict) else user_input
    conversations[user_id].append(HumanMessage(content=input_message_content))
    conversations[user_id] = conversations[user_id][-10:] # Keep the last 10 messages for context
    print(f"Chat: User '{user_id}' input: '{input_message_content}'")
    print(f"Chat: Current conversation history length: {len(conversations[user_id])}")

    # Invoke the LangGraph with the full conversation history
    print("Chat: Invoking LangGraph for non-streaming response...")
    result = await graph.ainvoke({"messages": conversations[user_id]})
    print("Chat: LangGraph invocation completed.")
    
    # Get the latest AI reply from the result
    ai_reply = result["messages"][-1]
    
    # Store the AI's reply in the conversation history
    conversations[user_id].append(ai_reply)
    print(f"Chat: AI reply stored. Content length: {len(ai_reply.content)}")
    
    return {"reply": ai_reply.content}

@app.post("/chat-stream")
async def chat_stream(request: Request):
    print("\n--- Received /chat-stream request ---")
    body = await request.json()
    user_id = body.get("user_id", "default")
    user_input = body.get("message")
    # Retrieve the optional mock_data from the request body
    mock_data = body.get("mock_data")

    if not user_input:
        print("Chat-Stream: Missing 'message' in request body.")
        return {"error": "Missing 'message' in request body."}

    input_message_content = json.dumps(user_input) if isinstance(user_input, dict) else user_input
    conversations[user_id].append(
        HumanMessage(content=input_message_content)
    )
    conversations[user_id] = conversations[user_id][-10:]
    print(f"Chat-Stream: User '{user_id}' input: '{input_message_content}'")
    print(f"Chat-Stream: Current conversation history length: {len(conversations[user_id])}")

    async def token_generator():
        print("🚀 Chat-Stream: Token generator started.")
        
        full_reply_content = ""
        
        # Create a shared queue for tokens
        token_queue = asyncio.Queue()
        print("Chat-Stream: Token queue created.")
        
        async def stream_callback(token: str):
            """Callback function to receive tokens from the LLM"""
            # This is called by StreamCaptureHandler's on_llm_new_token
            print(f"📤 Chat-Stream: Callback received token: {repr(token)[:50]}...") # Print a snippet
            await token_queue.put(token)
        
        # Create a task to run the graph
        async def run_graph():
            try:
                print("🔄 Chat-Stream: Starting graph execution task...")
                
                # Construct the input for the graph.
                graph_input = {
                    "messages": conversations[user_id], 
                    "stream_callback": stream_callback
                }
                
                # Add mock_data to the graph input if it was provided in the request
                if mock_data:
                    graph_input["mock_data"] = mock_data
                
                # Run the graph with the stream callback
                result = await graph.ainvoke(graph_input)
                
                print("✅ Chat-Stream: Graph execution task completed.")
                
                # Signal end of stream
                await token_queue.put(None)
                
                # After graph execution, if the AIMessage was returned by abdurrahman_node
                # it will be in result["messages"]. We then update the conversation history.
                final_ai_message = next((m for m in reversed(result["messages"]) if m.type == "ai"), None)
                if final_ai_message:
                    # This adds the complete AI message to the session's conversation history
                    # ensuring future turns have the full context.
                    # It's crucial for the graph node (abdurrahman_node) to return the AIMessage
                    # as part of its state update.
                    if final_ai_message.content == full_reply_content:
                        # Only append if not already appended by the `token_generator` loop
                        # This avoids duplicates if `full_reply_content` already includes it
                        pass # The logic below will handle appending after stream
                    else:
                        # This branch should ideally not be hit if abdurrahman_node returns
                        # the full content and token_generator also reconstructs it.
                        # For now, let's rely on token_generator's full_reply_content for storage.
                        print("Chat-Stream: WARNING: Mismatch between final_ai_message from graph and accumulated stream content.")


            except Exception as e:
                print(f"🔥 Chat-Stream: Error in graph execution task: {e}")
                import traceback
                traceback.print_exc() # Print full traceback
                await token_queue.put(None)
        
        # Start the graph execution in the background
        graph_task = asyncio.create_task(run_graph())
        print("Chat-Stream: Graph execution task scheduled.")
        
        try:
            # Yield tokens as they come from the queue
            while True:
                # Wait for token with timeout to avoid hanging
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=60.0) # Increased timeout
                except asyncio.TimeoutError:
                    print("⚠️ Chat-Stream: Timeout waiting for token from queue. This indicates an issue upstream.")
                    break
                
                if token is None:  # End of stream signal
                    print("🏁 Chat-Stream: End of stream signal received.")
                    break
                
                full_reply_content += token
                
                # Format as Server-Sent Events
                yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.001) # Reduced sleep for faster streaming
            
            # Wait for graph task to truly complete, catching any final exceptions
            await graph_task
            
            # Store the complete response in conversation history
            if full_reply_content.strip():
                # Avoid appending duplicate if `AIMessage` was already added by `abdurrahman_node`
                # We need to check if the last message in history is the one we just built
                if not conversations[user_id] or conversations[user_id][-1].content != full_reply_content:
                    conversations[user_id].append(AIMessage(content=full_reply_content))
                    print(f"💾 Chat-Stream: Stored complete response: {len(full_reply_content)} characters.")
                else:
                    print("💾 Chat-Stream: Complete response already in history (deduplicated).")
            else:
                print("Chat-Stream: No content generated for storing.")
            
            # Send end signal to client
            yield f"data: {json.dumps({'done': True})}\n\n"
            print("Chat-Stream: Sent 'done' signal.")
            
        except Exception as e:
            print(f"🔥 Chat-Stream: ERROR in token_generator loop: {e}")
            import traceback
            traceback.print_exc() # Print full traceback
            yield f"data: {json.dumps({'error': f'An error occurred during streaming: {str(e)}'})}\n\n"
        
        print("✅ Chat-Stream: Stream processing completed for this request.")

    return StreamingResponse(
        token_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )
