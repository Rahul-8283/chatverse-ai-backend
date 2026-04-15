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
        list: The generated embedding vector (768 dimensions for text-embedding-004).
    """
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
            alt_model = 'models/embedding-001' if not EMBEDDING_MODEL.startswith('models/') else 'embedding-001'
            result = genai.embed_content(
                model=alt_model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']
            print(f"✅ Generated embedding with {len(embedding)} dimensions using model: {alt_model}")
            return embedding
    except Exception as e:
        print(f"❌ Error generating embedding for text: {text[:100]}...")
        print(f"Exception: {e}")
        raise

def generate_query_embedding(text: str) -> list:
    """
    Generates an embedding for a user query.

    Args:
        text (str): The query text to embed.

    Returns:
        list: The generated embedding vector for the query (768 dimensions for text-embedding-004).
    """
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
            alt_model = 'models/embedding-001' if not EMBEDDING_MODEL.startswith('models/') else 'embedding-001'
            result = genai.embed_content(
                model=alt_model,
                content=text,
                task_type="retrieval_query"
            )
            embedding = result['embedding']
            print(f"✅ Generated query embedding with {len(embedding)} dimensions using: {alt_model}")
            return embedding
    except Exception as e:
        print(f"❌ Error generating query embedding: {e}")
        raise
