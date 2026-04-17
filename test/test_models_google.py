"""
Quick test to check available embedding models
"""
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# List available models
print("Available models:")
for model in genai.list_models():
    print(f"  - {model.name}")
    if "embed" in model.name.lower():
        print(f"    ✅ EMBEDDING MODEL FOUND: {model.name}")
