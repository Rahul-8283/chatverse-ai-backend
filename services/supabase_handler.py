"""
Handles file storage operations with Supabase Storage.
"""
from supabase import create_client, Client
import os
from .config import SUPABASE_URL, SUPABASE_SECRET_KEY, SUPABASE_BUCKET

# --- Supabase Client Initialization ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    print("Supabase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    raise

def upload_file_to_storage(user_id: str, file_path: str, file_name: str) -> str:
    """
    Uploads a file to a user-specific folder in the Supabase bucket.

    Args:
        user_id (str): The unique ID of the user.
        file_path (str): The local path to the file to be uploaded.
        file_name (str): The name of the file as it will be stored.

    Returns:
        str: The public URL of the uploaded file.
    
    Raises:
        Exception: If the file upload fails.
    """
    storage_path = f"{user_id}/{file_name}"
    
    try:
        with open(file_path, 'rb') as f:
            # Upload the file
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": "application/octet-stream"} # Generic content type
            )
        
        # Get the public URL of the uploaded file
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
        
        print(f"File '{file_name}' uploaded to Supabase at path: {storage_path}")
        return public_url

    except Exception as e:
        print(f"Error uploading file to Supabase: {e}")
        # Here you might want to check if the error is because the file already exists
        # and handle it, e.g., by updating the file or returning the existing URL.
        # For now, we re-raise the exception.
        raise Exception(f"Supabase upload failed: {e}")


def delete_file_from_storage(user_id: str, file_name: str) -> bool:
    """
    Deletes a file from a user-specific folder in the Supabase bucket.

    Args:
        user_id (str): The unique ID of the user.
        file_name (str): The name of the file to be deleted.

    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    storage_path = f"{user_id}/{file_name}"
    
    try:
        response = supabase.storage.from_(SUPABASE_BUCKET).remove([storage_path])
        print(f"Attempted to delete file from Supabase: {storage_path}")
        # The response for a successful deletion is usually a list with one item.
        # We can check if the response indicates success.
        if response:
            return True
        return False
    except Exception as e:
        print(f"Error deleting file from Supabase: {e}")
        return False
