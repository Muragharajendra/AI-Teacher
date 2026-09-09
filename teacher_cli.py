
from Backend.teacher.graph import graph


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

