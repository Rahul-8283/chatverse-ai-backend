"""
Handles processing of uploaded files, such as extracting text from PDFs
and chunking text into smaller pieces.
"""
import PyPDF2
from io import BytesIO
import speech_recognition as sr
import google.generativeai as genai
from PIL import Image

from .config import GENERATIVE_MODEL

async def extract_text_from_image(file_content: bytes) -> str:
    """
    Generates a description for an image using Gemini.

    Args:
        file_content (bytes): The byte content of the image file.

    Returns:
        str: The generated text description.
    """
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        image = Image.open(BytesIO(file_content))
        
        prompt = "Describe this image in detail. This description will be used for a search index, so be comprehensive."
        
        response = await model.generate_content_async([prompt, image])
        print("Generated image description with Gemini.")
        return response.text
    except Exception as e:
        print(f"Error generating image description: {e}")
        raise

def extract_text_from_audio(file_content: bytes, file_name: str) -> str:
    """
    Transcribes audio content to text using SpeechRecognition library.
    It handles WAV format directly. For other formats like MP3 or WEBM,
    conversion might be needed beforehand (requires ffmpeg).

    Args:
        file_content (bytes): The byte content of the audio file.
        file_name (str): The name of the file, used to infer format.

    Returns:
        str: The transcribed text.
    """
    recognizer = sr.Recognizer()
    
    # The recognizer needs an AudioFile object
    # We can use BytesIO to treat the byte content as a file
    try:
        with sr.AudioFile(BytesIO(file_content)) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"Transcribed audio file '{file_name}'.")
            return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        print("Please ensure the audio is in a compatible format (like WAV) and ffmpeg is installed for other formats.")
        raise


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extracts text from the content of a PDF file.

    Args:
        file_content (bytes): The byte content of the PDF file.

    Returns:
        str: The extracted text.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        print(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """
    Splits a long text into smaller, overlapping chunks.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum size of each chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    
    print(f"Split text into {len(chunks)} chunks.")
    return chunks
