"""
Handles the generation of embeddings using the Gemini API.
"""
import google.generativeai as genai
from .config import GEMINI_API_KEY, EMBEDDING_MODEL
import time
import random

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    raise

def generate_embedding(text: str, max_retries: int = 5) -> list:
    """
    Generates an embedding for a single piece of text with exponential backoff retry.

    Args:
        text (str): The text to embed.
        max_retries (int): Maximum number of retry attempts (default: 5).

    Returns:
        list: The generated embedding vector (768 dimensions for text-embedding-004).
    """
    for attempt in range(max_retries):
        try:
            # Try the primary model
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = result['embedding']
                print(f"✅ Generated embedding with {len(embedding)} dimensions using model: {EMBEDDING_MODEL}")
                return embedding
            except Exception as e1:
                # Fallback to alternative model format
                print(f"⚠️ {EMBEDDING_MODEL} failed, trying alternative format...")
                alt_model = 'models/gemini-embedding-001' if not EMBEDDING_MODEL.startswith('models/') else 'gemini-embedding-001'
                result = genai.embed_content(
                    model=alt_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = result['embedding']
                print(f"✅ Generated embedding with {len(embedding)} dimensions using model: {alt_model}")
                return embedding
        except Exception as e:
            # Check if it's a rate limit error (429)
            if "429" in str(e) or "Resource exhausted" in str(e):
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds + random jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Rate limited (429). Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Max retries exceeded. Failed after {max_retries} attempts.")
                    raise
            else:
                print(f"❌ Error generating embedding for text: {text[:100]}...")
                print(f"Exception: {e}")
                raise
    
    raise Exception("Failed to generate embedding after all retry attempts")

def generate_query_embedding(text: str, max_retries: int = 5) -> list:
    """
    Generates an embedding for a user query with exponential backoff retry.

    Args:
        text (str): The query text to embed.
        max_retries (int): Maximum number of retry attempts (default: 5).

    Returns:
        list: The generated embedding vector for the query (768 dimensions for text-embedding-004).
    """
    for attempt in range(max_retries):
        try:
            # Try the primary model
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_query"
                )
                embedding = result['embedding']
                print(f"✅ Generated query embedding with {len(embedding)} dimensions")
                return embedding
            except Exception as e1:
                # Fallback to alternative model format
                print(f"⚠️ {EMBEDDING_MODEL} failed for query, trying alternative format...")
                alt_model = 'models/gemini-embedding-001' if not EMBEDDING_MODEL.startswith('models/') else 'gemini-embedding-001'
                result = genai.embed_content(
                    model=alt_model,
                    content=text,
                    task_type="retrieval_query"
                )
                embedding = result['embedding']
                print(f"✅ Generated query embedding with {len(embedding)} dimensions using: {alt_model}")
                return embedding
        except Exception as e:
            # Check if it's a rate limit error (429)
            if "429" in str(e) or "Resource exhausted" in str(e):
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds + random jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Rate limited (429). Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Max retries exceeded. Failed after {max_retries} attempts.")
                    raise
            else:
                print(f"❌ Error generating query embedding: {e}")
                raise
    
    raise Exception("Failed to generate query embedding after all retry attempts")
