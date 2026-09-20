from Backend.doc_parser.text_process import INP_pdf
from Backend.doc_parser.TOC_gen_LLM import LLM_TOC_GEN
from Backend.retrieval.run_retrieval import initialize_retrievers
from Backend.teacher.graph import get_graph

import asyncio

from Backend.doc_parser.text_process import INP_pdf
from Backend.doc_parser.TOC_gen_LLM import LLM_TOC_GEN
from Backend.retrieval.run_retrieval import initialize_retrievers
import sys
from pathlib import Path          # only if you actually use it elsewhere
from Backend.teacher.graph import get_graph
from Backend.voice.tts import speak_streaming
from Backend.voice.stt import listen_once

async def pdf_process_async():

    # Run ONLY these two functions concurrently
    await asyncio.gather(
        asyncio.to_thread(LLM_TOC_GEN),
        asyncio.to_thread(initialize_retrievers, True)
    )
    print("================LLM TOC JSON GENERATED.================")
    print("========== Retrieval system initialized. ==========")

def pdf_process(extraction, file):  # Extraction=True when new docs uploaded.
    if extraction:
        INP_pdf(file)
        print("================Text Extracted from PDF================")

        # Run LLM_TOC_GEN() and initialize_retrievers(True) in parallel
        asyncio.run(pdf_process_async())

    else:
        # Existing vector store
        initialize_retrievers(False)

NEW_DOC_Status = False  # received from frontend if user uploads new docs and press continue.
                       # User does this once and first. after pressing continue he will go to sec step where he does chating, only retrieving
file="Backend/docs/inp_docs/NCERT-Class-10-History.pdf"
pdf_process(NEW_DOC_Status, file) 
# NOTE: NEW DOC UPLOAD is True make it False after first run, and make True again if you uploAd new document.

print("=========chunking, Retrieval starts=========")
# Teacher Output
# this part should run when click to continue on voice mode
def print_teacher_response(result):
    response = result.get("response")
    print("\nTeacher:")
    print(response if response else "[No response returned]")
    print()

# main function should run again if user change the mode
def main(mode):
    # ----------------------------------------------------------------
    # Mode selection — "chat" or "voice"
    # ----------------------------------------------------------------
    mode = mode          # change to "chat" this will be recieved from frontend 
    # mode = sys.argv[1].lower() if len(sys.argv) > 1 else "voice"

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

    # ----------------------------------------------------------------
    # Main conversation loop
    # ----------------------------------------------------------------
    while True:
        query = ""                       # explicit, so it's never stale/unbound

        # ── 1. Acquire input ────────────────────────────────
        try:
            if mode == "voice":
                print("\n🎤 Listening...")
                query = listen_once()
                if not query:
                    print("No speech detected.")
                    continue
                print(f"\nStudent: {query}")
            else:
                query = input("Student: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        # ── 2. Run the graph (Ctrl-C safe) ──────────────────
        try:
            result = graph.invoke(
                {"user_query": query, "mode": mode},
                config=config,
            )
        except KeyboardInterrupt:
            print("\nInterrupted — exiting...")
            break
        except Exception as e:
            print(f"\n[GRAPH ERROR] {type(e).__name__} - {e}\n")
            continue

        print_teacher_response(result)

        # ── 3. Voice reply (Ctrl-C safe) ────────────────────
        if mode == "voice":
            response_text = result.get("response", "")
            if response_text:
                try:
                    speak_streaming(response_text)
                except KeyboardInterrupt:
                    print("\n[Interrupted during TTS]")
                    break
                except Exception as e:
                    print(f"[TTS failed: {type(e).__name__}: {e}]")
if __name__ == "__main__":
    mode="voice"
    conversation=True
    if conversation:
        main(mode)