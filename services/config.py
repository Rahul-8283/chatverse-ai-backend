"""
Configuration module to load and manage environment variables.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Gemini AI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# --- Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("Pinecone API Key or Index Name environment variable not set.")

# --- Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
if not SUPABASE_URL or not SUPABASE_SECRET_KEY or not SUPABASE_BUCKET:
    raise ValueError("Supabase URL, Secret Key, or Bucket Name environment variable not set.")

# --- Firebase Admin ---
# For service account authentication, we'll load the credentials from the environment variables
# The private key needs special handling to replace newline characters
FIREBASE_PRIVATE_KEY = os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n')

FIREBASE_CREDENTIALS = {
    "type": "service_account",
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"), # This is optional but good practice
    "private_key": FIREBASE_PRIVATE_KEY,
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"), # Optional
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('FIREBASE_CLIENT_EMAIL')}"
}

# --- Model Configuration ---
EMBEDDING_MODEL = 'gemini-embedding-001'
EMBEDDING_DIMENSION = 3072  # gemini-embedding-001 produces 3072-dimensional vectors
GENERATIVE_MODEL = 'gemini-2.5-flash'
GROQ_MODEL = 'llama-3.1-70b-versatile'  # Groq model for fallback

