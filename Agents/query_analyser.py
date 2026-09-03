import json

query="Ernst Renan, ‘What is a Nation'"

with open("docs/TOC_from_llm/TOC_from_llm_1.json", "r", encoding="utf-8") as f:
    TOC_from_llm=json.load(f)
# print("TOC_from_llm:", TOC_from_llm)
