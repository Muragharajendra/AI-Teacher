import os
from dotenv import load_dotenv
from openai import OpenAI
from text_process import text_extract
import json

# Input for llm to get proper TOC
text_extracted=text_extract()
if not text_extracted.strip():
    raise ValueError("Extracted Text Not Found")

load_dotenv() 

# Check api load
api_key=os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key not Found")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
with open("docs/promt_to_get_TOC.txt", "r", encoding="utf-8") as f:
    promt=f.read()
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": promt
        },
        {
            "role": "user",
            "content": text_extracted
        }
    ],
    temperature=0,
    max_completion_tokens=4096
)
try:
    result=response.choices[0].message.content
    result = response.choices[0].message.content.strip()

    if result.startswith("```json"):
        result = result[7:]

    if result.startswith("```"):
        result = result[3:]

    if result.endswith("```"):
        result = result[:-3]

    result = result.strip()
    parsed=json.loads(result)
    with open("docs/TOC_from_llm/TOC_from_llm_1.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=4, ensure_ascii=False )
    print("Json created successfully")
except json.JSONDecodeError:
    print(result)
    print("IMproper data. Cant creat TOC JSON")