import os
from dotenv import load_dotenv
from openai import OpenAI
from text_process import text_extract_for_llm
import json

# Input for llm to get proper TOC
text_extracted=text_extract_for_llm()
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

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": """ You are an expert document structure extraction engine.
            Your task is to reconstruct the hierarchical structure of a textbook from two inputs:
            1. Table of Contents (TOC)
            2. Headings extracted from the document
            Rules:
            1. Use the TOC ONLY to determine:
            - Section names
            - Chapter names
            - Their order
            2. Use the extracted headings ONLY to determine:
            - Sections inside each chapter
            - Subsections inside each section
            - Their order
            3. Never invent, infer, summarize, rewrite, rename, or merge different headings.
            4. Every chapter appearing in the TOC MUST appear in the output, even if no headings were detected for it.
            In that case, use an empty object.
            5. Preserve the exact wording of headings.
            Do not change capitalization, punctuation, spelling, or numbering.
            8. Remove duplicate headings while preserving the first occurrence.
            9. Maintain the original hierarchy exactly as it appears in the document.
            10. Do not create additional hierarchy levels that are not present.
            11. If a heading cannot be confidently classified, attach it to the nearest valid parent heading.
            12. The output must be deterministic.
                Do not guess missing information.
            Return ONLY valid JSON.
            JSON schema:
            {
            "Section Name": {
                "Chapter Name": {
                "Section": {
                    "Subsection": {}
                }
                }
            }
            }
            Return ONLY the JSON object.
            Do not include markdown.
            Do not use ```json.
            Do not include explanations.
            Do not include notes."""
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