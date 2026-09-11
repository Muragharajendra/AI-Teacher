from Backend.doc_parser.text_process import INP_pdf
from Backend.doc_parser.TOC_gen_LLM import LLM_TOC_GEN
from Backend.retrieval.run_retrieval import initialize_retrievers
from Backend.teacher.graph import graph

import asyncio

from Backend.doc_parser.text_process import INP_pdf
from Backend.doc_parser.TOC_gen_LLM import LLM_TOC_GEN
from Backend.retrieval.run_retrieval import initialize_retrievers


async def pdf_process_async():

    # Run ONLY these two functions concurrently
    await asyncio.gather(
        asyncio.to_thread(LLM_TOC_GEN),
        asyncio.to_thread(initialize_retrievers, True)
    )
    print("================LLM TOC JSON GENERATED.================")
    print("========== Retrieval system initialized. ==========")


def pdf_process(extraction):  # Extraction=True when new docs uploaded.
    if extraction:
        INP_pdf("Backend/docs/inp_docs/NCERT-Class-10-History.pdf")
        print("================Text Extracted from PDF================")

        # Run LLM_TOC_GEN() and initialize_retrievers(True) in parallel
        asyncio.run(pdf_process_async())

    else:
        # Existing vector store
        initialize_retrievers(False)

NEW_DOC_Status = True  # received from frontend if user uploads new docs and press continue.
                       # User does this once and first. after pressing continue he will go to sec step where he does chating, only retrieving
pdf_process(NEW_DOC_Status) 
# NOTE: NEW DOC UPLOAD is True make it False after first run, and make True again if you uploAd new document.

print("=========chunking, Retrieval starts=========")
# Teacher Output
def print_teacher_response(result):
    """Print the teacher's response."""

    response = result.get("response")

    print("\nTeacher:")
    if response:
        print(response)
    else:
        print("[No response returned]")
    print()


def main():
    # --------------------------------------------------
    # SESSION
    # --------------------------------------------------

    thread_id = "student-001"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print("\n========================================")
    print("             AI TEACHER")
    print("========================================")
    print(f"Thread ID: {thread_id}")
    print("Type 'exit' or 'quit' to stop.\n")

    # --------------------------------------------------
    # MAIN CONVERSATION LOOP
    # --------------------------------------------------

    while True:

        try:
            query = input("Student: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        # --------------------------------------------------
        # Ignore empty input
        # --------------------------------------------------

        if not query:
            continue

        # --------------------------------------------------
        # Exit
        # --------------------------------------------------

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:

            # --------------------------------------------------
            # SEND QUERY TO LANGGRAPH
            # --------------------------------------------------

            result = graph.invoke(
                {
                    "user_query": query
                },
                config=config
            )

            # --------------------------------------------------
            # PRINT RESPONSE
            # --------------------------------------------------

            print_teacher_response(result)

        except Exception as e:

            print("\nERROR:")
            print(type(e).__name__, "-", e)
            print()


if __name__ == "__main__":
    main()

                