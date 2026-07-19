def text_preprocess(text):
    # for line in text:
    #     if not line:
    #         continue
    #     line=line.strip()
    pass

    

with open("docs/extracted_text/text_md_1.md","r",encoding="utf-8") as infile:
     inp_text=infile.readlines()

text_preprocess(inp_text)