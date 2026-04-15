import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import RAG services and authentication
from services import rag_service
from services.firebase_auth import verify_firebase_token
from services.config import GEMINI_API_KEY, GENERATIVE_MODEL

# --- FastAPI App Initialization ---
app = FastAPI(title="ChatVerse AI Backend")

# --- CORS Middleware ---
# Allows frontend to communicate with this backend
origins = [
    "http://localhost:5173",
    "https://chatverse-ai.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully for main app.")
except Exception as e:
    print(f"Error configuring Gemini API in main app: {e}")
    raise

# --- Pydantic Models for Request Bodies ---
class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    persona: str

class RAGChatRequest(BaseModel):
    query: str
    persona: Optional[str] = "default"

# --- API Endpoints ---

@app.get("/", summary="Root endpoint to check server status")
async def root():
    """Returns a welcome message and server status."""
    return {"message": "ChatVerse AI Backend is running!"}

# --- Standard Chat Endpoint (Direct to Gemini) ---
@app.post("/api/chat", summary="Handles direct chat with Gemini AI")
async def chat_handler(request: ChatRequest):
    """
    Receives a chat message and history, and returns a response from Gemini.
    This endpoint does NOT use the RAG pipeline.
    """
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        
        # Constructing the chat history for the model
        chat_history = []
        for item in request.history:
            # Skip empty items
            if not item or not isinstance(item, dict):
                continue
            # Check if required fields exist
            if "sender" not in item or "text" not in item:
                continue
            
            role = "user" if item["sender"] == "user" else "model"
            chat_history.append({"role": role, "parts": [{"text": item["text"]}]})
            
        chat = model.start_chat(history=chat_history)
        response = await chat.send_message_async(request.message)
        
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- RAG Chat Endpoint ---
@app.post("/api/rag-chat", summary="Handles chat queries using the RAG pipeline")
async def rag_chat_handler(request: RAGChatRequest, user_id: str = Depends(verify_firebase_token)):
# async def rag_chat_handler(request: RAGChatRequest, user_id: str = "test-user"):  # For testing - change back to Depends(verify_firebase_token)

    """
    Receives a query and uses the RAG service to generate a context-aware response.
    This endpoint is secured and requires a valid Firebase ID token.
    """
    try:
        result = await rag_service.rag_query(user_id=user_id, query=request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Document Upload Endpoint ---
@app.post("/api/upload-document", summary="Uploads and processes a document for RAG")
async def upload_document_handler(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_firebase_token)
    # user_id: str = "test-user"  # For testing - change back to Depends(verify_firebase_token) later

):
    """
    Handles uploading of a document (PDF, image, audio).
    The document is processed and its embeddings are stored in Pinecone.
    This endpoint is secured.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")
    
    try:
        file_content = await file.read()
        content_type = file.content_type
        
        # Process and store the document in the background
        # For a real production app, you would use a task queue like Celery or ARQ
        await rag_service.process_and_store_document(
            user_id=user_id,
            file_content=file_content,
            file_name=file.filename,
            file_type=content_type
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "message": "File is being processed. It will be available for queries shortly."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {e}")

# --- Image Scan Endpoint (Existing, could be adapted for RAG) ---
@app.post("/api/image-scan", summary="Analyzes an uploaded image")
async def image_scan_handler(file: UploadFile = File(...), prompt: str = Form(...)):
    """
    Analyzes an image using Gemini's multimodal capabilities.
    This could be extended to store image descriptions as embeddings for RAG.
    """
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        
        # Ensure file is an image
        if "image" not in file.content_type:
            raise HTTPException(status_code=400, detail="File must be an image.")
            
        image_bytes = await file.read()
        image_parts = [{"mime_type": file.content_type, "data": image_bytes}]
        
        prompt_parts = [prompt, image_parts[0]]
        
        response = await model.generate_content_async(prompt_parts)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Voice Processing Endpoint (Existing, could be adapted for RAG) ---
@app.post("/api/voice", summary="Processes a voice recording")
async def voice_handler(file: UploadFile = File(...)):
    """
    A placeholder for voice processing. This would be where speech-to-text
    is performed before chunking and embedding for RAG.
    """
    # In a real implementation, you would use a speech-to-text library here.
    # For now, we'll just return a placeholder.
    return {"response": "Voice processing not yet implemented. The transcript would be embedded here."}
