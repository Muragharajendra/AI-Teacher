import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from text_process import text_extract
import json



# Testing
# with open("docs/extracted_text/text_md_test.md", "r", encoding="utf-8") as f:
#     text_extracted=f.read()


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# Check api load
api_key=os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key not Found")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)


def LLM_TOC_GEN():
    # Input for llm to get proper TOC
    text_extracted=text_extract()
    if not text_extracted.strip():
        raise ValueError("Extracted Text Not Found")

    with open("docs/promt_to_get_TOC.txt", "r", encoding="utf-8") as f:
        promt=f.read()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
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
        with open("Backend/docs/Final_LLM_responses/TOC_from_llm_1.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=4, ensure_ascii=False )
        # print("Json created successfully")
    except json.JSONDecodeError:
        raise ValueError("Improper data. Cant create TOC JSON")
        print(result)
        print("Improper data. Cant create TOC JSON")