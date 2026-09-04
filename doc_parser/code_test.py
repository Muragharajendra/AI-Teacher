import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

print("API key loaded:", api_key[:8] + "...")

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is a ball? Explain in simple words."
            }

        ],
        temperature=0.5,
        max_completion_tokens=1024
    )

    print("CONTENT:")
    print(repr(response.choices[0].message.content))

    print("REASONING:")
    print(repr(response.choices[0].message.reasoning))

    print("FINISH:")
    print(response.choices[0].finish_reason)

except Exception as e:
    print("\n❌ Error:")
    print(e)