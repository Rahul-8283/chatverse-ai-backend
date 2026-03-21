# ChatVerse AI Backend

This is the FastAPI backend for the ChatVerse AI application. It provides endpoints for chat, image analysis, and voice transcription using the Google Gemini 2.5 Flash model.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`.
   ```bash
   cp .env.example .env
   ```

3. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will be available at `http://localhost:8000`.

## API Documentation

### POST `/api/chat`
Handles conversational AI responses with persona selection.
* **Request Body (JSON):**
  ```json
  {
    "message": "Hello!",
    "history": [],
    "persona": "assistant"
  }
  ```
* **Response (JSON):**
  ```json
  {
    "reply": "Hi! How can I help you today?"
  }
  ```

### POST `/api/image-scan`
Analyzes an uploaded image and extracts text using Gemini Vision.
* **Request:** Multipart Form Data (`file`: Image blob, e.g. jpg, png, webp)
* **Response (JSON):**
  ```json
  {
    "reply": "Image description from Gemini..."
  }
  ```

### POST `/api/voice`
Transcribes an uploaded audio file using Gemini.
* **Request:** Multipart Form Data (`file`: Audio blob, e.g. webm, wav)
* **Response (JSON):**
  ```json
  {
    "transcript": "Transcribed audio text..."
  }
  ```

### GET `/health`
Health check endpoint.
* **Response (JSON):**
  ```json
  {
    "status": "online"
  }
  ```

## Deployment on Railway
1. Push your code to a GitHub repository.
2. Link your repository in Railway.
3. Add the `GEMINI_API_KEY` to the service variables in Railway.
4. Railway will automatically detect the Python environment, install `requirements.txt`, and run `uvicorn main:app --host 0.0.0.0 --port $PORT` if a Procfile or configuration is automatically generated (if not, specify the start command manually: `uvicorn main:app --host 0.0.0.0 --port $PORT`).