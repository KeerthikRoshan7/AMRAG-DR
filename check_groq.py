import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hi"}]
    )
    print("Groq is connected!")
except Exception as e:
    print(f"Connection failed: {e}")