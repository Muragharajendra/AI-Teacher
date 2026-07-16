import re
import pymupdf4llm

doc="docs/inp_docs/NCERT-Class-10-History.pdf"

text_mark=pymupdf4llm.to_markdown(doc, header=False, footer=False)
with open("docs/extracted_text/text_md_1.md", "w", encoding="utf-8") as f:
    f.write(text_mark)

REMOVE_KEYWORDS = {
    "activity",
    "discuss",
    "project",
    "write in brief",
    "new words",
    "source",
    "box",
    "fig.",
    "figure",
    "case study",
    "self-study",
    "text book",
    "reference book",
    "suggested learning resources",
    "web links",
    "assessment pattern",
    "marks distribution",
}
def text_extract_for_llm():
    # with open("docs/text_md_1.md", "r", encoding="utf-8") as infile, \
        #  open("docs/processed_text_for_llm/process_text_for_llm_1.md", "w") as outfile:
    
    text_llm=[] # IMP # reset automatically every function call
    patterns = [
        re.compile(r'^#{1,6}\s+.+$'),                          # Markdown headings
        re.compile(r'^MODULE[- ]?\d+.*$'),                     # MODULE-1
        re.compile(r'^CHAPTER\s+\d+.*$'),                      # CHAPTER 1
        re.compile(r'^Chapter\s+\d+.*$'),                      # Chapter 1
        re.compile(r'^[IVXLCDM]+\.\s+.+$'),                    # I. Introduction
        re.compile(r'^\d+(\.\d+)*\s+.+$'), 
                        
    ]
    for line_num, line in enumerate(text_mark.splitlines(), start=1):
        if not line:
            continue
        if any(pattern.match(line) for pattern in patterns):
            # outfile.write("["+str(line_num)+"]"+" "+line.strip()+ "\n")
            text_llm.append("["+str(line_num)+"]"+" "+line.strip()+ "\n")
    a=[]
    for line in text_llm:
        lower_line=line.lower()
        if any(keyword in lower_line for keyword in REMOVE_KEYWORDS):
            continue
        a.append(line)
    return "\n".join(a)
