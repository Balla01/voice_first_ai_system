import os
from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()

# Read API key
api_key = os.getenv("groq_api")

if not api_key:
    raise ValueError("groq_api not found in .env")

# Create client
client = Groq(api_key=api_key)

# Create completion
completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Explain AI in simple terms"
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
)

# Stream response
for chunk in completion:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")