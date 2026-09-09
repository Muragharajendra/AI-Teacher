from Backend.doc_parser.text_process import INP_pdf
from Backend.doc_parser.TOC_gen_LLM import LLM_TOC_GEN
from teacher_cli import main

INP_pdf("Backend/docs/inp_docs/NCERT-Class-10-History.pdf")
print("================Text Extracted from PDF================")

LLM_TOC_GEN()
print("================LLM TOC JSON GENERATED.================")

print("=========chunking Retrieval starts=========")
main()                     # Rebuild is True now make it False after first run, and make True again if you uplod new document.