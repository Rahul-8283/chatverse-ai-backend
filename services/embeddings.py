"""
Handles the generation of embeddings using the Gemini API.
"""
import google.generativeai as genai
from .config import GEMINI_API_KEY, EMBEDDING_MODEL

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    raise

def generate_embedding(text: str) -> list:
    """
    Generates an embedding for a single piece of text.

    Args:
        text (str): The text to embed.

    Returns:
        list: The generated embedding vector.
    """
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document" # Use "retrieval_query" for queries
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding for text: {text[:100]}...")
        print(f"Exception: {e}")
        raise

def generate_query_embedding(text: str) -> list:
    """
    Generates an embedding for a user query.

    Args:
        text (str): The query text to embed.

    Returns:
        list: The generated embedding vector for the query.
    """
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise
