"""
Orchestrates the entire Retrieval-Augmented Generation (RAG) pipeline.
"""
import uuid
import time
from . import data_processor, embeddings, pinecone_handler, supabase_handler
import google.generativeai as genai
from groq import Groq
from .config import GENERATIVE_MODEL, GROQ_API_KEY, GROQ_MODEL
from firebase_admin import firestore

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Lazy initialization of Firestore client
_db = None

def get_db():
    """Get Firestore client with lazy initialization."""
    global _db
    if _db is None:
        _db = firestore.client()
    return _db

async def rag_query(user_id: str, query: str) -> dict:
    """
    Performs a RAG query with a fallback from Gemini to Groq.
    
    1. Generates an embedding for the user's query.
    2. Queries Pinecone to find relevant context.
    3. Builds a prompt with the context and query.
    4. Calls the generative model to get a response, with Groq as fallback.
    """
    print(f"Performing RAG query for user '{user_id}': '{query}'")
    
    # 1. Generate query embedding
    query_embedding = embeddings.generate_query_embedding(query)
    
    # 2. Query Pinecone for relevant context
    matches = pinecone_handler.query_vectors(
        query_embedding=query_embedding,
        top_k=3,
        namespace=user_id
    )
    
    # 3. Build context from matches
    context = "Context from user's documents:\n"
    sources = []
    if matches:
        for match in matches:
            # Ensure metadata and its fields exist
            if 'metadata' in match and 'chunk_text' in match['metadata'] and 'file_name' in match['metadata']:
                context += f"- {match['metadata']['chunk_text']}\n"
                sources.append({
                    "file_name": match['metadata']['file_name'],
                    "chunk_text": match['metadata']['chunk_text']
                })
    else:
        context = "No relevant context found in user's documents."

    # 4. Build the final prompt
    prompt = f"""
    You are a helpful AI assistant. Answer the user's query based on the provided context.
    If the context is not sufficient, use your general knowledge but state that the information
    is not from the user's documents.

    {context}

    User Query: {query}
    
    Answer:
    """
    
    # 5. Call the generative model with Gemini-to-Groq fallback
    response_text = ""
    
    # Try Gemini first
    try:
        print("Attempting RAG query with Gemini...")
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        response = await model.generate_content_async(prompt)
        response_text = response.text
        print("✅ Successfully generated RAG response with Gemini.")
    except Exception as e:
        print(f"❌ Gemini RAG failed: {e}. Falling back to Groq.")
        
        # Fallback to Groq
        try:
            print("Attempting RAG query with Groq...")
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Answer based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
                ],
                model=GROQ_MODEL,
            )
            response_text = chat_completion.choices[0].message.content
            print("✅ Successfully generated RAG response with Groq.")
        except Exception as e_groq:
            print(f"❌ Groq RAG also failed: {e_groq}")
            raise Exception("Both Gemini and Groq RAG failed.")
    
    return {
        "response": response_text,
        "sources": sources,
        "used_rag": bool(matches)  # True if context was found and used
    }
