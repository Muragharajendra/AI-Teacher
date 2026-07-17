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

        Your task is to reconstruct the hierarchical structure of a textbook using two inputs:

        Table of Contents (TOC)
        Extracted document headings

        The goal is to generate a complete and deterministic hierarchical JSON representation of the textbook.

        Input Definitions
        Table of Contents (TOC)

        The TOC is the authoritative source for:

        Section names
        Chapter names
        Chapter numbering
        Section ordering
        Chapter ordering
        Printed book page numbers

        The printed page number appearing after a chapter title (for example, The Making of a Global World 77) is the chapter's book page number and should be included in the output.

        Extracted Headings

        The extracted headings are the authoritative source for:

        Sections within chapters
        Subsections within sections
        Any deeper heading hierarchy
        Their original order

        Use extracted headings only to build the hierarchy inside each chapter.

        Rules
        Use the TOC only to determine the textbook's top-level structure.
        Use extracted headings only to determine the internal hierarchy of each chapter.
        Every chapter listed in the TOC must appear in the output, even if no headings are found for that chapter.
        If no headings belong to a chapter, return an empty "sections" object.
        Preserve the exact wording, capitalization, punctuation, numbering, and spelling of every heading.
        Never invent, infer, summarize, rename, merge, or rewrite headings.
        Remove duplicate headings while preserving only their first occurrence.
        Preserve the original document order exactly.
        Do not reorder headings alphabetically.
        Do not create hierarchy levels that do not exist in the document.
        If a heading cannot be confidently classified, attach it to the nearest valid parent heading.
        Ignore body text, paragraphs, page numbers, headers, footers, running headers, watermarks, and other non-heading content.
        Ignore headings that do not belong to any TOC chapter.
        The output must be deterministic. Given the same input, always produce identical JSON.
        Never guess or fabricate missing information.
        Output Format

        Return only valid JSON.

        Do not include:

        Markdown
        Code fences
        Comments
        Notes
        Explanations
        Any text outside the JSON object

        Use the following schema:

        {
        "Section Name": {
        "chapters": {
        "Chapter Name": {
        "book_page": 77,
        "sections": {
        "Section Heading": {
        "Subsection Heading": {}
        }
        }
        }
        }
        }
        }

        If a chapter has no detected headings:

        {
        "Section Name": {
        "chapters": {
        "Chapter Name": {
        "book_page": 77,
        "sections": {}
        }
        }
        }
        }

        Return only the JSON object."""
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