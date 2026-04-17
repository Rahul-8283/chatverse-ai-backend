"""
Quick test to check available Groq models
"""
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# List available models
print("Available Groq models:")
models = groq_client.models.list()
for model in models.data:
    print(f"  - {model.id}")
    if hasattr(model, 'type'):
        print(f"    Type: {model.type}")
    if hasattr(model, 'description'):
        print(f"    Description: {model.description}")
