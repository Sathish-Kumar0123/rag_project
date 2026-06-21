import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List all available models
models = genai.list_models()

print("\nAvailable Models:\n")
for m in models:
    print(m.name, " | Supported:", m.supported_generation_methods)