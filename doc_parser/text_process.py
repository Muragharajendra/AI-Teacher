import re
import pymupdf4llm
# from pymupdf4llm.ocr import tesseract_api

DOC = "docs/inp_docs/NCERT-Class-10-History.pdf"

pages = pymupdf4llm.to_markdown(
    DOC,
    header=False,
    footer=True,
    page_chunks=True,

    # OCR
    use_ocr=False,
    # ocr_language="eng",
    # ocr_dpi=300,
    # ocr_function=tesseract_api,

    show_progress=True
)

extracted_text=[]

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
patterns = [
        re.compile(r'^#{1,6}\s+.+$'),                          # Markdown headings
        re.compile(r'^MODULE[- ]?\d+.*$'),                     # MODULE-1
        re.compile(r'^CHAPTER\s+\d+.*$'),                      # CHAPTER 1
        re.compile(r'^Chapter\s+\d+.*$'),                      # Chapter 1
        re.compile(r'^[IVXLCDM]+\.\s+.+$'),                    # I. Introduction
        re.compile(r'^\d+(\.\d+)*\s+.+$'), 
                        
    ]
MARKDOWN_FOOTER_PATTERNS = [
    re.compile(r"^\d+$"),                                  # 12
    re.compile(r"^[-–—\s]*\d+[-–—\s]*$"),                  # - 12 -
    re.compile(r"^[\*_]*(page|p\.)\s*-?\s*\d+[\*_]*$", re.IGNORECASE),  # **Page 12** or *p. 12*
    re.compile(r"^[\*_]*\d+\s*(of|/)\s*\d+[\*_]*$", re.IGNORECASE)  # **12 of 50** or _12/50_
]

for chunk in pages:
    # page_num=chunk["metadata"]["page_number"]
    # extracted_text.append(f"page number:{page_num}")

    lines = chunk["text"].splitlines()
    for line in lines:
        if line.strip():
            extracted_text.append(line)   

extracted_info="\n".join(extracted_text)

with open("docs/extracted_text/text_md_1.md", "w", encoding="utf-8") as f:
    ext_lines=extracted_info.splitlines()
    for line_ind, line in enumerate(ext_lines):
        if any(patt.match(line) for patt in MARKDOWN_FOOTER_PATTERNS):
            f.write(f"footer page number:{line.strip()}\n")
        elif line_ind < len(ext_lines) - 1 and "footer page number:" in ext_lines[line_ind] and "footerpage number:" in ext_lines[line_ind + 1]:
            continue
        else:
            f.write(f"{line}\n")
            
        
        

def text_extract_for_llm():
    # with open("docs/text_md_1.md", "r", encoding="utf-8") as infile, \
        #  open("docs/processed_text_for_llm/process_text_for_llm_1.md", "w") as outfile:
    
    text_llm=[] # IMP # reset automatically every function call
    for line_num, line in enumerate(extracted_info.splitlines(), start=1):
        clean_line=line.strip()
        if not clean_line:
            continue
        if any(patt.match(clean_line) for patt in MARKDOWN_FOOTER_PATTERNS):
            text_llm.append(f"[{line_num}] footer page number:{clean_line}\n")
        elif any(pattern.match(clean_line) for pattern in patterns):
            # outfile.write("["+str(line_num)+"]"+" "+clean_line+ "\n")
            text_llm.append(f"[{line_num}] {clean_line}\n")
    # text1_llm=[]
    # for i in text_llm:
    #     if not i.strip():

    a=[]
    for line_ind, line in enumerate(text_llm):
        lower_line = line.lower()
        if "footer page number:" in lower_line:

            if line_ind < len(text_llm) - 1 and "footer page number:" in text_llm[line_ind].lower() and "footer page number:" in text_llm[line_ind + 1].lower():
                continue
            a.append(line)
            continue
        if any(keyword in lower_line for keyword in REMOVE_KEYWORDS):
            continue
        if not line.strip():
            continue
        a.append(line)
    return "\n".join(a)

def text_extract():
    list1 = []
    ex_text = text_extract_for_llm()
    lines = ex_text.splitlines()
    
    for line_index, i in enumerate(lines):
        clean_line = i.strip().lower()
        
        if "footer page number:" in clean_line:
            is_duplicate = False
            for next_idx in range(line_index + 1, len(lines)):
                next_line_stripped = lines[next_idx].strip().lower()

                if "footer page number:" in next_line_stripped:
                    is_duplicate = True
                    break
                elif next_line_stripped:
                    break
            if is_duplicate:
                continue # Skip the first footer
                
        list1.append(i)
        
    return "\n".join(list1)
with open("docs/extracted_text/text_md_test.md", "w", encoding="utf-8") as f:
     f.write(text_extract())
print("Done!")