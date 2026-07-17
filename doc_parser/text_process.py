import re
import pymupdf4llm

doc="docs/inp_docs/NCERT-Class-10-History.pdf"

chunks=pymupdf4llm.to_markdown(doc, header=False, footer=False, page_chunks=True)
extracted_text=[]

for chunk in chunks:
    page_num=chunk["metadata"]["page_number"]
    extracted_text.append(f"page number:{page_num}")

    lines = chunk["text"].splitlines()
    for line in lines:
        if line.strip():
            extracted_text.append(line)   

extracted_info="\n".join(extracted_text)

with open("docs/extracted_text/text_md_1.md", "w", encoding="utf-8") as f:
    ext_lines=extracted_info.splitlines()
    for line_ind, line in enumerate(ext_lines):
        if line_ind < len(ext_lines) - 1 and "page number:" in ext_lines[line_ind] and "page number:" in ext_lines[line_ind + 1]:
                    continue
        f.write(f"{line}\n")

REMOVE_KEYWORDS = {
    "page number:",
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
    
    for line_num, line in enumerate(extracted_info.splitlines(), start=1):
        if not line:
            continue
        if any(pattern.match(line) for pattern in patterns) or "page number:" in line:
            # outfile.write("["+str(line_num)+"]"+" "+line.strip()+ "\n")
            text_llm.append("["+str(line_num)+"]"+" "+line.strip()+ "\n")
    a=[]
    for line_ind, line in enumerate(text_llm):
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in REMOVE_KEYWORDS):
            continue
        # if line_ind < len(text_llm) - 1 and "page number:" in text_llm[line_ind] and "page number:" in text_llm[line_ind + 1]:
        #     continue
        a.append(line)
    return "\n".join(a)
print(text_extract_for_llm())