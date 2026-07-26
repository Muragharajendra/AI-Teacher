from preprocessing_extracted_text import text_preprocess, open_file

# files
text, inp_txt= open_file()

all_chapters=text_preprocess(text, inp_txt)  # chapter wise chunked list

# print(all_chapters[2])
