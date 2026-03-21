<div align="center">

# 🤖 ChatVerse AI - Core Backend  
**A blazingly fast FastAPI intelligent routing engine for text, voice, and multimodal Gemini 2.5 context handling.**

![Python Badge](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) 
![FastAPI Badge](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white) 
![Gemini Badge](https://img.shields.io/badge/Powered_by-Gemini_2.5_Flash-orange?logo=google&logoColor=white)
![Render Deploy](https://img.shields.io/badge/Render-Deployment-blueviolet)

</div>

<br/>

## 🌐 Overview
The **ChatVerse AI Backend** acts as the crucial middle-layer between the high-performance React UI and Google's Gemini LLMs. Built completely in asynchronous Python using `FastAPI` and `uvicorn`, this server securely handles API keys, dictates context limits, manages dynamic System Prompts (Personas), and parses multipart payloads natively so the frontend doesn't have to decode binary assets.

## 🏗️ Architecture & Workflow Diagram

![ChatVerse AI Workflow](https://app.eraser.io/workspace/7qSCXI4BZomyTXpyUvn1/preview)

---

## ⚡ Features Overview

### The AI Routing Layer
This backend acts strictly as an API gateway to Google Gemini, keeping the main React application lightweight and secure:
- **Persona Context Engineering**: Dynamically swaps system prompts (Assistant, Roast Bot, Therapist, Study Buddy) before reaching the models.
- **Multimodal Image Pipeline**: Securely processes raw image byte uploads and maps them to Gemini Vision text extractors natively.
- **Voice Transcription Muxxing**: Parses `.webm` audio streams natively from web recorders, translates them in real-time, and responds contextually.
- **Access Portability**: Protects the `GEMINI_API_KEY` server-side, preventing browser scraping or local injection exploits.

*(Note: Standard person-to-person messaging does not route through this backend code. P2P chat relies purely on direct Firebase connections in the frontend.)*

---

## 🚀 Quick Start (Local Development)

**1. Clone & Install**
```bash
git clone https://github.com/Rahul-8283/chatverse-ai-backend.git
cd chatverse-ai-backend
pip install -r requirements.txt
```

**2. Configure Environment**
Create a `.env` file at the root containing your Gemini key:
```dotenv
GEMINI_API_KEY=AIzaSy...your_gemini_key_here
```

**3. Boot the Server**
```bash
uvicorn main:app --reload
```
*Your interactive API documentation is now live at `http://localhost:8000/docs`!*

---

## 📚 API Architecture Routing

### 1. Unified Text Chat (`POST /api/chat`)
Expects standard JSON histories and handles the persona mappings globally:
```json
// Request
{
  "message": "Explain black holes",
  "history": [{"role": "user", "parts": [{"text":"hello"}]}],
  "persona": "study"
}
```

### 2. Vision Intelligence (`POST /api/image-scan`)
Accepts raw multipart `FormData` intercepting browser `FileReader` boundaries flawlessly:
* **Payload:** `file: <Blob>` (jpg, png, webp)
* **Response:** `{ "reply": "This image contains..." }`

### 3. Audio Extraction (`POST /api/voice`)
Intercepts HTML5 MediaRecorder chunks securely:
* **Payload:** `file: <Blob>` (audio/webm)
* **Response:** `{ "transcript": "What the user actually said." }`