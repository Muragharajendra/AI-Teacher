from Backend.doc_parser.text_process import INP_pdf
from Backend.doc_parser.TOC_gen_LLM import LLM_TOC_GEN
from Backend.retrieval.run_retrieval import initialize_retrievers
from Backend.teacher.graph import get_graph

import asyncio

from Backend.doc_parser.text_process import INP_pdf
from Backend.doc_parser.TOC_gen_LLM import LLM_TOC_GEN
from Backend.retrieval.run_retrieval import initialize_retrievers
import sys


async def pdf_process_async():

    # Run ONLY these two functions concurrently
    await asyncio.gather(
        # asyncio.to_thread(LLM_TOC_GEN),
        asyncio.to_thread(initialize_retrievers, True)
    )
    print("================LLM TOC JSON GENERATED.================")
    print("========== Retrieval system initialized. ==========")

def pdf_process(extraction):  # Extraction=True when new docs uploaded.
    if extraction:
        # INP_pdf("Backend/docs/inp_docs/NCERT-Class-10-History.pdf")
        print("================Text Extracted from PDF================")

        # Run LLM_TOC_GEN() and initialize_retrievers(True) in parallel
        asyncio.run(pdf_process_async())

    else:
        # Existing vector store
        initialize_retrievers(False)

NEW_DOC_Status = False  # received from frontend if user uploads new docs and press continue.
                       # User does this once and first. after pressing continue he will go to sec step where he does chating, only retrieving
pdf_process(NEW_DOC_Status) 
# NOTE: NEW DOC UPLOAD is True make it False after first run, and make True again if you uploAd new document.

print("=========chunking, Retrieval starts=========")
# Teacher Output
def print_teacher_response(result):
    response = result.get("response")
    print("\nTeacher:")
    print(response if response else "[No response returned]")
    print()


def main():
    # mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "chat")
    mode = "chat"
    if mode not in ("chat", "voice"):
        print("Usage: python -m Backend.teacher.teacher_cli [chat|voice]")
        return

    graph = get_graph(mode)

    thread_id = f"student-001-{mode}"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n========================================")
    print(f"             AI TEACHER  ({mode} mode)")
    print("========================================")
    print(f"Thread ID: {thread_id}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Student: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            result = graph.invoke({"user_query": query, "mode": mode}, config=config)
            print_teacher_response(result)
        except Exception as e:
            print(f"\nERROR:\n{type(e).__name__} - {e}\n")


if __name__ == "__main__":
    main()

                