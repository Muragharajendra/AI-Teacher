from preprocessing_extracted_text import text_preprocess, open_file
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# files
ext_text, inp_txt= open_file()

all_chapters=text_preprocess(ext_text, inp_txt)  # chapter wise chunked list

# writing first in file text_md_test_1.md
with open("docs/extracted_text/text_md_test_1.md", "w", encoding="utf-8") as f:
    f.write(all_chapters[0])  # writing first chapter in file
# print(all_chapters[0])

def create_chunks(all_chapters):

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Chapter Name"),
            ("##", "subheading1"),
            ("###", "subheading2"),
            ("####", "subheading3"),
            ("#####", "subheading4"),
            ("######", "subheading5"),
            ("#######", "subheading6"),
            ("########", "subheading7")
        ],
        strip_headers=False
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    final_chunks = []

    for chapter in all_chapters:

        # chapter should be your markdown-formatted chapter text
        sections = markdown_splitter.split_text(chapter)

        for section in sections:

            # If section is already small, don't split it
            if len(section.page_content) <= 600:
                final_chunks.append(section)

            # If section is large, recursively split it
            else:
                smaller_chunks = recursive_splitter.split_documents(
                    [section]
                )

                final_chunks.extend(smaller_chunks)
        
    # for chunk in final_chunks:
    #     print("CHunk meta data:", chunk.metadata) 
    #     print("chunk content:", chunk.page_content)
    #     print("="*100)

    print(f"Total number of chunks created: {len(final_chunks)}")
    # print(final_chunks[0])

final_chunks = create_chunks(all_chapters)