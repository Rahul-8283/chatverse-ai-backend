"""
Handles vector database operations with Pinecone.
"""
from pinecone import Pinecone, ServerlessSpec
from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# --- Pinecone Initialization ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if the index exists. If not, create it.
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating new index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Gemini embeddings dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Pinecone index created successfully.")
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone client and index initialized successfully.")

except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    raise

def upsert_vectors(vectors: list, namespace: str):
    """
    Upserts (inserts or updates) vectors into the Pinecone index.

    Args:
        vectors (list): A list of vectors to upsert. Each vector should be a dict
                        with 'id', 'values', and 'metadata'.
        namespace (str): The namespace to upsert the vectors into (e.g., user_id).
    """
    try:
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Upserted {len(vectors)} vectors into namespace '{namespace}'.")
    except Exception as e:
        print(f"Error upserting vectors to Pinecone: {e}")
        raise

def query_vectors(query_embedding: list, top_k: int, namespace: str) -> list:
    """
    Queries the Pinecone index to find the most similar vectors.

    Args:
        query_embedding (list): The embedding of the user's query.
        top_k (int): The number of similar results to return.
        namespace (str): The namespace to search within (e.g., user_id).

    Returns:
        list: A list of matching documents from the query result.
    """
    try:
        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        print(f"Query returned {len(results['matches'])} matches from namespace '{namespace}'.")
        return results['matches']
    except Exception as e:
        print(f"Error querying vectors from Pinecone: {e}")
        raise

def delete_namespace(namespace: str):
    """
    Deletes all vectors within a specific namespace.

    Args:
        namespace (str): The namespace to delete (e.g., user_id).
    """
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted all vectors in namespace '{namespace}'.")
    except Exception as e:
        print(f"Error deleting namespace from Pinecone: {e}")
        raise
