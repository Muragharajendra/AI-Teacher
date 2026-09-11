import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from Backend.doc_parser.text_process import text_extract
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

    with open(BASE_DIR /"Backend/docs/promt_to_get_TOC.txt", "r", encoding="utf-8") as f:
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
        result = response.choices[0].message.content.strip()

        if result.startswith("```json"):
            result = result[7:]

        if result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()
        if not result:
            raise ValueError("LLM returned an empty response")
        print("LLM_TOC_Result", result)
        
        parsed=json.loads(result)
        with open(BASE_DIR /"Backend/docs/Final_LLM_responses/TOC_from_llm_1.json", "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=4, ensure_ascii=False )
                # print("Json created successfully")
    except json.JSONDecodeError as e:
        print("========== JSON ERROR ==========")
        print("Error:", e)
        print("Raw LLM result:")
        print(repr(result))
        print("================================")
        raise ValueError("LLM returned invalid JSON") from e
    
if __name__=="__main__":
    LLM_TOC_GEN()