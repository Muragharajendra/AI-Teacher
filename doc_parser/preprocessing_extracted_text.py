import json

# Reading TOC JSON file and Extracted text file
with open("docs/extracted_text/text_md_1.md","r",encoding="utf-8") as infile_01, \
     open("docs/TOC_from_llm/TOC_from_llm_1.json","r",encoding="utf-8") as infile_02:  
        inp_text=infile_01.readlines()
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
    
    # Extracting chapter page number range
    def recursive_search(self, obj):
        ranges = []
        if isinstance(obj, dict):
            # Check if current dictionary contains chapter page markers
            if "start_chapter_page" in obj and "end_chapter_page" in obj:
                ranges.append([obj["start_chapter_page"], obj["end_chapter_page"]])
            # Recurse into nested dictionaries
            for value in obj.values():
                ranges.extend(self.recursive_search(value))
        elif isinstance(obj, list):
            for item in obj:
                ranges.extend(self.recursive_search(item))
        return ranges

TOC_d_obj=TOC_details(inp_txt) 
# print(TOC_d_obj.chapter_count())
# print(TOC_d_obj.recursive_search(inp_txt))
        
# Extracting text chunking
def text_preprocess(text):
    end_chapter=[]
    
    chapters=[]
    chapter_count=TOC_d_obj.chapter_count()
    chapter_range=TOC_d_obj.recursive_search(inp_txt)
    for page in chapter_range:
        end_chapter.append(page[1])
    start_line=0
    for i in end_chapter:
        chapter_l=[]
        for line in range(start_line, len(text)):
            if f"footer page number:{i+1}" not in text[line]:
                chapter_l.append(text[line])
            else:
                start_line=line+1
                break
        chapter_info="\n".join(chapter_l)
        chapters.append(chapter_info)
    print(chapters[2])



    
    # for _ in range(chapter_count):  # spliting chapter wiss
    #     # print(ranges)
    #     pass
        

text_preprocess(inp_text)
