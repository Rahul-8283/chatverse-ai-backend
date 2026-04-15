"""
Orchestrates the entire Retrieval-Augmented Generation (RAG) pipeline.
"""
import uuid
from . import data_processor, embeddings, pinecone_handler, supabase_handler
import google.generativeai as genai
from .config import GENERATIVE_MODEL

async def process_and_store_document(user_id: str, file_content: bytes, file_name: str, file_type: str):
    """
    Processes an uploaded document, generates embeddings, and stores them.
    
    1. Uploads the file to Supabase storage.
    2. Extracts text from the document based on its type (PDF, image, audio).
    3. Chunks the text.
    4. Generates embeddings for each chunk.
    5. Upserts the embeddings into Pinecone.
    """
    print(f"\n🚀 Starting document processing for user '{user_id}', file '{file_name}', type '{file_type}'.")
    
    # 0. Upload file to Supabase
    try:
        print(f"☁️ Uploading file to Supabase...")
        import tempfile
        import os as os_module
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os_module.path.splitext(file_name)[1]) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        # Upload to Supabase
        file_url = supabase_handler.upload_file_to_storage(user_id, tmp_path, file_name)
        print(f"✅ File uploaded to Supabase: {file_url}")
        
        # Clean up temp file
        os_module.remove(tmp_path)
    except Exception as e:
        print(f"⚠️ Warning: Could not upload to Supabase (non-critical): {e}")
        # Continue processing even if upload fails - still store in Pinecone
    
    text = ""
    # 1. Extract text based on file type
    if file_type == 'application/pdf':
        print("📥 Extracting text from PDF...")
        text = data_processor.extract_text_from_pdf(file_content)
    elif file_type.startswith('image/'):
        print("🖼️ Generating description from image...")
        text = await data_processor.extract_text_from_image(file_content)
    elif file_type.startswith('audio/'):
        print("🎤 Transcribing audio...")
        text = data_processor.extract_text_from_audio(file_content, file_name)
    else:
        raise ValueError(f"❌ Unsupported file content type: {file_type}")

    # 2. Chunk the text
    chunks = data_processor.chunk_text(text)
    
    if not chunks:
        print("⚠️ No text chunks to process.")
        return

    # 3. Generate embeddings for each chunk
    print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
    chunk_embeddings = [embeddings.generate_embedding(chunk) for chunk in chunks]
    
    # 4. Prepare vectors for Pinecone
    print("📝 Preparing vectors for Pinecone...")
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        vector_id = str(uuid.uuid4())
        vectors_to_upsert.append({
            "id": vector_id,
            "values": chunk_embeddings[i],
            "metadata": {
                "user_id": user_id,
                "file_name": file_name,
                "chunk_text": chunk[:500]  # Store first 500 chars for reference
            }
        })

    # 5. Upsert to Pinecone
    try:
        pinecone_handler.upsert_vectors(vectors=vectors_to_upsert, namespace=user_id)
        print(f"✅ Successfully stored {len(vectors_to_upsert)} vectors in Pinecone for user '{user_id}'")
    except Exception as e:
        print(f"❌ Failed to upsert vectors to Pinecone: {e}")
        raise
    
    print("✅ Document processing and storage complete.\n")



async def rag_query(user_id: str, query: str) -> dict:
    """
    Performs a RAG query.
    
    1. Generates an embedding for the user's query.
    2. Queries Pinecone to find relevant context.
    3. Builds a prompt with the context and query.
    4. Calls the generative model to get a response.
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
    
    # 5. Call the generative model
    model = genai.GenerativeModel(GENERATIVE_MODEL)
    response = await model.generate_content_async(prompt)
    
    return {
        "response": response.text,
        "sources": sources,
        "used_rag": bool(matches) # True if context was found and used
    }
