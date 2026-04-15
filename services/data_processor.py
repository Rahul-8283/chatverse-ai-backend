"""
Handles processing of uploaded files, such as extracting text from PDFs
and chunking text into smaller pieces.
"""
import PyPDF2
from io import BytesIO

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
