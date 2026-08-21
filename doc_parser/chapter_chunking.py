from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from doc_parser.markdown_cleaner import clean_and_normalize_markdown

# Read cleaned markdown from markdown_cleaner output
with open("docs/extracted_text/text_md_1.md", "r", encoding="utf-8") as f:
        raw_markdown = f.read()
markdown_text=clean_and_normalize_markdown(raw_markdown)  # cleaning will be done in markdown_cleaner.py

def create_chunks(markdown_text):

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
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", " ", ""]
    )

    final_chunks = []

    # Process entire markdown text directly - no chapter iteration needed
    sections = markdown_splitter.split_text(markdown_text)
    for section in sections:
        # If section is already small, don't split it
        if len(section.page_content) <= 600:
            final_chunks.append(section)
        # If section is large, recursively split it
        else:
            smaller_chunks = recursive_splitter.split_documents([section])
            final_chunks.extend(smaller_chunks)
        
    # for chunk in final_chunks:
    #     print("CHunk meta data:", chunk.metadata) 
    #     print("chunk content:", chunk.page_content)
    #     print("="*100)

    print(f"Total number of chunks created: {len(final_chunks)}")
    # print(final_chunks[0])
    
    return final_chunks

final_chunks = create_chunks(markdown_text)

# Write chunks to text_md_test_1.md
with open("docs/extracted_text/text_md_test_1.md", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(final_chunks):
        f.write(f"--- CHUNK {i} ---\n")
        f.write(f"Metadata: {chunk.metadata}\n")
        f.write(f"Content:\n{chunk.page_content}\n")
        f.write("="*100 + "\n\n")

print(f"\n All {len(final_chunks)} chunks written to docs/extracted_text/text_md_test_1.md")