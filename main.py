import os
import google.generativeai as genai
from groq import Groq
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import RAG services and authentication
from services import rag_service, data_processor, chat_service, document_service
from services.firebase_auth import verify_firebase_token
from services.config import GEMINI_API_KEY, GROQ_API_KEY, GENERATIVE_MODEL, GROQ_MODEL

# --- FastAPI App Initialization ---
app = FastAPI(title="ChatVerse AI Backend")

# --- CORS Middleware ---
# Allows frontend to communicate with this backend
origins = [
    "http://localhost:5173",
    "https://chatverse-ai-chat.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Client Configuration ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # We don't raise here to allow fallback to work if Gemini key is missing

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq API client configured successfully.")
except Exception as e:
    print(f"Error configuring Groq API client: {e}")
    # We don't raise here to allow fallback to work if Groq key is missing

# --- Pydantic Models for Request Bodies ---
class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    persona: str

class RAGChatRequest(BaseModel):
    query: str

# --- API Endpoints ---

@app.get("/health", summary="Root endpoint to check server status")
async def root():
    """Returns a welcome message and server status."""
    return {"message": "ChatVerse AI Backend is running!"}

# --- Standard Chat Endpoint (Direct to LLM with Fallback) ---
@app.post("/api/chat", summary="Handles direct chat with an LLM, with fallback")
async def chat_handler(request: ChatRequest):
    """
    Receives a chat message and history, and returns a response from an LLM.
    It first tries Gemini and falls back to Groq on specific errors.
    """
    # Constructing the chat history for the models
    chat_history = []
    for item in request.history:
        if not item or "sender" not in item or "text" not in item:
            continue
        role = "user" if item["sender"] == "user" else "assistant"
        chat_history.append({"role": role, "content": item["text"]})

    # 1. Try Gemini first
    try:
        print("Attempting to generate content with Gemini...")
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        
        # Reformat history for Gemini
        gemini_history = [
            {"role": "user" if h["role"] == "user" else "model", "parts": [{"text": h["content"]}]}
            for h in chat_history
        ]
        
        chat = model.start_chat(history=gemini_history)
        response = await chat.send_message_async(request.message)
        print("Successfully generated content with Gemini.")
        return {"response": response.text}
    except Exception as e:
        print(f"Gemini API failed: {e}. Falling back to Groq.")

    # 2. Fallback to Groq
    try:
        print("Attempting to generate content with Groq...")
        # Add the current user message to the history for Groq
        chat_history.append({"role": "user", "content": request.message})
        
        # Add system prompt for persona
        if request.persona:
            chat_history.insert(0, {"role": "system", "content": f"You are {request.persona}. Respond accordingly."})

        chat_completion = groq_client.chat.completions.create(
            messages=chat_history,
            model=GROQ_MODEL,
        )
        response_text = chat_completion.choices[0].message.content
        print("Successfully generated content with Groq.")
        return {"response": response_text}
    except Exception as e_groq:
        print(f"Groq API also failed: {e_groq}")
        raise HTTPException(status_code=500, detail="Both Gemini and Groq APIs failed.")

# --- RAG Chat Endpoint ---
@app.post("/api/rag-chat", summary="Handles chat queries using the RAG pipeline")
async def rag_chat_handler(request: RAGChatRequest, user_id: str = Depends(verify_firebase_token)):
# async def rag_chat_handler(request: RAGChatRequest, user_id: str = "test-user"):  # For testing

    """
    Receives a query and uses the RAG service to generate a context-aware response.
    This endpoint is secured and requires a valid Firebase ID token.
    """
    try:
        # The fallback logic is now inside rag_service.rag_query
        result = await rag_service.rag_query(
            user_id=user_id, 
            query=request.query
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Document Upload Endpoint ---
@app.post("/api/upload-document", summary="Uploads and processes a document for RAG")
async def upload_document_handler(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_firebase_token)
    # user_id: str = "test-user"  # For testing

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
        
        await document_service.process_and_store_document(
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

# --- Image Scan Endpoint (with Fallback) ---
@app.post("/api/image-scan", summary="Analyzes an uploaded image with fallback")
async def image_scan_handler(file: UploadFile = File(...), prompt: str = Form(...)):
    """
    Analyzes an image using Gemini's multimodal capabilities.
    This does not fall back to Groq as Groq does not support image inputs.
    """
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        
        if "image" not in file.content_type:
            raise HTTPException(status_code=400, detail="File must be an image.")
            
        image_bytes = await file.read()
        image_parts = [{"mime_type": file.content_type, "data": image_bytes}]
        
        prompt_parts = [prompt, image_parts[0]]
        
        response = await model.generate_content_async(prompt_parts)
        return {"response": response.text}
    except Exception as e:
        print(f"Image scan with Gemini failed: {e}")
        raise HTTPException(status_code=500, detail="Image analysis failed. The fallback API does not support images.")

# --- Voice Processing Endpoint ---
@app.post("/api/voice", summary="Processes a voice recording")
async def voice_handler(file: UploadFile = File(...)):
    """
    Transcribes audio to text using speech-to-text.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided.")
    
    try:
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail=f"File must be an audio file, got {file.content_type}")
        
        file_content = await file.read()
        
        transcript = data_processor.extract_text_from_audio(file_content, file.filename)
        
        return {
            "success": True,
            "transcript": transcript,
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

# --- Document Management Endpoints ---
@app.get("/api/documents", summary="Get all documents uploaded by user")
async def get_documents(user_id: str = Depends(verify_firebase_token)):
    """
    Retrieves all documents uploaded by the current user from Firestore.
    """
    try:
        documents = await document_service.get_user_documents(user_id)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

@app.delete("/api/documents/delete-all", summary="Delete all documents for user")
async def delete_all_documents(user_id: str = Depends(verify_firebase_token)):
    """
    Deletes all documents uploaded by the user from Firestore, Supabase, and Pinecone.
    """
    try:
        await document_service.delete_all_documents(user_id)
        return {"success": True, "message": "All documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all documents: {str(e)}")

@app.delete("/api/documents/{doc_id}", summary="Delete a specific document")
async def delete_document(doc_id: str, user_id: str = Depends(verify_firebase_token)):
    """
    Deletes a specific document from Firestore, Supabase, and Pinecone.
    """
    try:
        await document_service.delete_document(user_id, doc_id)
        return {"success": True, "message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# --- Chat Management Endpoints ---
@app.delete("/api/chat/{conversation_id}", summary="Delete a chat conversation")
async def delete_chat(conversation_id: str, user_id: str = Depends(verify_firebase_token)):
    """
    Deletes a specific AI chat conversation and all its messages from Firestore.
    The conversation_id can be: 'assistant', 'rag-analysis', 'therapist', etc.
    """
    try:
        print(f"🗑️ Delete chat request for conversation: {conversation_id}, user: {user_id}")
        await chat_service.delete_chat_conversation(user_id, conversation_id)
        return {"success": True, "message": f"Chat conversation '{conversation_id}' deleted successfully"}
    except Exception as e:
        print(f"❌ Error in delete_chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")
