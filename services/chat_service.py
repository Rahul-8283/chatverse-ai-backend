"""
Handles general chat management operations (not RAG-related).
"""
from firebase_admin import firestore


def get_db():
    """Get Firestore client."""
    return firestore.client()


async def delete_chat_conversation(user_id: str, conversation_id: str):
    """
    Deletes a specific AI chat conversation and all its messages from Firestore.
    
    Args:
        user_id (str): The user ID who owns the conversation
        conversation_id (str): The conversation ID to delete (e.g., "assistant", "rag-analysis", etc.)
    """
    try:
        print(f"🗑️ Starting deletion of conversation '{conversation_id}' for user '{user_id}'")
        
        db = get_db()
        
        # Get reference to messages collection
        messages_ref = db.collection('ai-chats').document(user_id).collection('conversations').document(conversation_id).collection('messages')
        
        # Fetch all messages
        messages = messages_ref.stream()
        message_count = 0
        
        # Delete all messages
        for message_doc in messages:
            print(f"🗑️ Deleting message: {message_doc.id}")
            message_doc.reference.delete()
            message_count += 1
        
        print(f"📋 Deleted {message_count} messages from conversation '{conversation_id}'")
        
        # Delete the conversation document itself
        conversation_ref = db.collection('ai-chats').document(user_id).collection('conversations').document(conversation_id)
        print(f"🗑️ Deleting conversation document: {conversation_id}")
        conversation_ref.delete()
        
        print(f"✅ Successfully deleted conversation '{conversation_id}' completely for user {user_id}")
    except Exception as e:
        print(f"❌ Error deleting AI conversation: {e}")
        raise
