"""
Handles vector database operations with Pinecone.
"""
from pinecone import Pinecone, ServerlessSpec
from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION

# --- Pinecone Initialization ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if the index exists. If not, create it.
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"🔄 Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating new index with {EMBEDDING_DIMENSION} dimensions...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,  # Use the dimension from config (3072 for gemini-embedding-001)
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' created successfully with {EMBEDDING_DIMENSION} dimensions.")
    else:
        print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    print("✅ Pinecone client and index initialized successfully.")

except Exception as e:
    print(f"❌ Error initializing Pinecone: {e}")
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

async def delete_vectors(vector_ids: list, namespace: str = None):
    """
    Deletes specific vectors by their IDs.

    Args:
        vector_ids (list): List of vector IDs to delete.
        namespace (str): Optional namespace to delete from. If not provided, deletes globally.
    """
    try:
        if namespace:
            index.delete(ids=vector_ids, namespace=namespace)
            print(f"Deleted {len(vector_ids)} vectors from namespace '{namespace}'.")
        else:
            index.delete(ids=vector_ids)
            print(f"Deleted {len(vector_ids)} vectors globally.")
    except Exception as e:
        print(f"Error deleting vectors from Pinecone: {e}")
        raise
