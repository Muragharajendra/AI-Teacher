import sys
from Backend.teacher.graph import get_graph


def print_teacher_response(result):
    response = result.get("response")
    print("\nTeacher:")
    print(response if response else "[No response returned]")
    print()


def main():
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "chat")
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