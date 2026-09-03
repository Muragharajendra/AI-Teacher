from retrieval.pre_LLM import ret_chunks  # Passing query and INP(Type) to pre_LLM.py for retrieval 
import json

query="Explain Ernst Renan What is a Nation concept in simple words"

with open("docs/TOC_from_llm/TOC_from_llm_1.json", "r", encoding="utf-8") as f:
    TOC_from_llm=json.load(f)
# print("TOC_from_llm:", TOC_from_llm)


# Based on the query, determine the appropriate retrieval method (INP)