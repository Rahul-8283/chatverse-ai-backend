import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chatverse-ai-chat.vercel.app",
        "http://localhost:5173",
        "https://chatverse-738ldwtdu-lsrahul12-6390s-projects.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class ChatPart(BaseModel):
    text: str

class ChatMessage(BaseModel):
    role: str
    parts: List[ChatPart]

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    persona: str = "assistant"

PERSONAS = {
    "assistant": "You are a helpful, friendly AI assistant.",
    "roast": "You are a savage but funny roast bot. Roast everything the user says. Keep it funny, not mean.",
    "study": "You are a patient study buddy. Explain concepts simply with examples. Ask follow-up questions to test understanding.",
    "therapist": "You are a calm, empathetic therapist. Listen carefully and respond with emotional intelligence and warmth."
}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    try:
        system_instruction = PERSONAS.get(request.persona, PERSONAS["assistant"])
        
        # Configure the model
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=system_instruction
        )

        # Reformat history if necessary
        formatted_history = []
        for msg in request.history:
            parts = [part.text for part in msg.parts]
            formatted_history.append({"role": msg.role, "parts": parts})
        
        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(request.message)
        
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/image-scan")
async def image_scan_endpoint(file: UploadFile = File(...), prompt: str = "Analyze and describe this image in detail."):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        contents = await file.read()
        
        image_parts = [
            {
                "mime_type": file.content_type,
                "data": contents
            }
        ]
        
        # 🔒 FIX #12: Use user's custom prompt instead of hardcoded one for multimodal support
        response = model.generate_content([prompt, image_parts[0]])
        
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        contents = await file.read()
        
        audio_parts = [
            {
                "mime_type": file.content_type,
                "data": contents
            }
        ]
        
        prompt = "Transcribe this audio message exactly as spoken."
        response = model.generate_content([prompt, audio_parts[0]])
        
        return {"transcript": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_endpoint():
    return {"status": "online"}
