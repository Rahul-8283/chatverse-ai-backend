"""
Handles document management operations (upload, retrieve, delete).
"""
import uuid
import time
from . import data_processor, embeddings, pinecone_handler, supabase_handler
from firebase_admin import firestore


def get_db():
    """Get Firestore client."""
    return firestore.client()


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

    # 3. Generate embeddings for each chunk with rate limiting
    print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            # Add 1 second delay between chunk processing to avoid rate limiting
            print(f"⏸️ Rate limiting pause before chunk {i + 1}/{len(chunks)}...")
            time.sleep(1)
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        embedding = embeddings.generate_embedding(chunk)
        chunk_embeddings.append(embedding)
    
    # 4. Prepare vectors for Pinecone
    print("📝 Preparing vectors for Pinecone...")
    vectors_to_upsert = []
    vector_ids = []  # Track all vector IDs for Firestore
    for i, chunk in enumerate(chunks):
        vector_id = str(uuid.uuid4())
        vector_ids.append(vector_id)
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
    
    # 6. Store document metadata in Firestore
    try:
        from datetime import datetime
        print("💾 Storing document metadata in Firestore...")
        
        doc_id = str(uuid.uuid4())
        user_docs_ref = get_db().collection('users').document(user_id).collection('documents')
        user_docs_ref.document(doc_id).set({
            'fileName': file_name,
            'fileType': file_type,
            'fileUrl': file_url if 'file_url' in locals() else '',
            'vectorIds': vector_ids,
            'uploadedAt': datetime.now().isoformat(),
            'chunkCount': len(chunks)
        })
        print(f"✅ Document metadata stored in Firestore with ID: {doc_id}")
    except Exception as e:
        print(f"❌ Failed to store document metadata in Firestore: {e}")
        # Don't raise - document is already in Pinecone
    
    print("✅ Document processing and storage complete.\n")


async def get_user_documents(user_id: str) -> list:
    """
    Retrieves all documents uploaded by a user from Firestore.
    Returns a list of document metadata including name, type, and id.
    """
    try:
        user_docs_ref = get_db().collection('users').document(user_id).collection('documents')
        docs = user_docs_ref.stream()
        
        documents = []
        for doc in docs:
            doc_data = doc.to_dict()
            documents.append({
                'id': doc.id,
                'name': doc_data.get('fileName', 'Unknown'),
                'type': doc_data.get('fileType', 'unknown'),
                'uploadedAt': doc_data.get('uploadedAt', ''),
                'vectorIds': doc_data.get('vectorIds', [])
            })
        
        return documents
    except Exception as e:
        print(f"❌ Error fetching user documents: {e}")
        raise


async def delete_document(user_id: str, doc_id: str):
    """
    Deletes a specific document from Firestore, Supabase, and Pinecone.
    """
    try:
        print(f"🔍 Starting deletion for document {doc_id} from user {user_id}")
        doc_ref = get_db().collection('users').document(user_id).collection('documents').document(doc_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise Exception(f"Document {doc_id} not found")
        
        doc_data = doc.to_dict()
        vector_ids = doc_data.get('vectorIds', [])
        file_name = doc_data.get('fileName', '')
        
        print(f"📋 Document data: fileName='{file_name}', vectorIds={vector_ids} (count: {len(vector_ids)})")
        
        # Delete from Pinecone
        if vector_ids:
            print(f"🗑️ Deleting {len(vector_ids)} vectors from Pinecone...")
            await pinecone_handler.delete_vectors(vector_ids, namespace=user_id)
            print(f"✅ Pinecone vectors deleted successfully")
        else:
            print(f"⚠️ No vectors found for this document - skipping Pinecone deletion")
        
        # Delete from Supabase if file name exists
        if file_name:
            try:
                print(f"🗑️ Deleting file from Supabase...")
                supabase_handler.delete_file_from_storage(user_id, file_name)
                print(f"✅ File deleted from Supabase")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete from Supabase: {e}")
        
        # Delete from Firestore
        print(f"🗑️ Deleting document metadata from Firestore...")
        doc_ref.delete()
        
        print(f"✅ Document {doc_id} deleted successfully")
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        raise


async def delete_all_documents(user_id: str):
    """
    Deletes all documents uploaded by a user from Firestore, Supabase, and Pinecone.
    """
    try:
        user_docs_ref = get_db().collection('users').document(user_id).collection('documents')
        docs = user_docs_ref.stream()
        
        doc_list = list(docs)
        print(f"🗑️ Deleting {len(doc_list)} documents for user {user_id}...")
        
        for doc in doc_list:
            await delete_document(user_id, doc.id)
        
        print(f"✅ All documents deleted successfully for user {user_id}")
    except Exception as e:
        print(f"❌ Error deleting all documents: {e}")
        raise
