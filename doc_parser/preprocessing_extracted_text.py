import json
import re

# Reading TOC JSON file and Extracted text file
with open("docs/extracted_text/text_md_1.md","r",encoding="utf-8") as infile_01, \
     open("docs/TOC_from_llm/TOC_from_llm_1.json","r",encoding="utf-8") as infile_02:  
        inp_text=infile_01.read()
        inp_txt=json.load(infile_02)

# TOC details extraction
class TOC_details():
    def __init__(self, txt):
        self.txt=txt
    def chapter_count(self):
        chapter_count=0
        patterns=["start_chapter_page"]
        str_lines=json.dumps(self.txt, indent=4)
        for line in str_lines.splitlines():
            if any(pattern in line for pattern in patterns):
                chapter_count+=1
        return chapter_count
    
    # Extracting chapter names
    def chapter_names(self, obj):
        chapters = []

        if isinstance(obj, dict):
            for key, value in obj.items():

                # Check if this key is a chapter
                if (
                    isinstance(value, dict)
                    and "start_chapter_page" in value
                    and "end_chapter_page" in value
                ):
                    chapters.append(key.lstrip("# ").strip())

                # Recurse into nested dictionaries
                chapters.extend(self.chapter_names(value))

        elif isinstance(obj, list):
            for item in obj:
                chapters.extend(self.chapter_names(item))

        return chapters

TOC_d_obj=TOC_details(inp_txt) 
# print(TOC_d_obj.chapter_count())
# print(TOC_d_obj.chapter_names(inp_txt))
        
# Extracting text chunking
def text_preprocess(text):
    end_chapter=[]
    
    chapters=[]
    # chapter_count=TOC_d_obj.chapter_count()
    all_chapter_names=TOC_d_obj.chapter_names(inp_txt)
    # print(all_chapter_names)
    start = 0
    chapters = []

    for chapter in all_chapter_names:
        pattern = rf"^#\s+(?:\*\*)?.*?{re.escape(chapter)}.*?(?:\*\*)?$"
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            end = match.start()
            chapters.append(text[start:end])
            start = end

    chapters.append(text[start:])
    for i in chapters:
        # print(i.splitlines()[0])
        pass
    print(chapters[7])



    
    # for _ in range(chapter_count):  # spliting chapter wiss
    #     # print(ranges)
    #     pass
        

text_preprocess(inp_text)
