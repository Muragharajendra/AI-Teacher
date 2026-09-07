from pathlib import Path

# Project root = folder where this script is located
BASE_DIR = Path(__file__).resolve().parent

# Folders/files to include
INCLUDE_PATHS = [
    BASE_DIR / "retrieval",
    BASE_DIR / "teacher",
    BASE_DIR / "teacher_cli.py",
]

OUTPUT_FILE = BASE_DIR / "doc_parser_retrieval_export.txt"

TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
}


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True

    try:
        path.read_text(encoding="utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def collect_files():
    files = []

    for path in INCLUDE_PATHS:
        if path.is_file():
            files.append(path)

        elif path.is_dir():
            for file in path.rglob("*"):
                if file.is_file():
                    files.append(file)

    # Remove duplicates and sort
    return sorted(set(files))


def create_export():
    files = collect_files()

    text_files = []
    binary_files = []

    for file in files:
        if is_text_file(file):
            text_files.append(file)
        else:
            binary_files.append(file)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:

        output.write("=" * 80 + "\n")
        output.write("PROJECT CODE EXPORT\n")
        output.write("=" * 80 + "\n\n")

        output.write(
            f"{len(text_files)} text files, "
            f"{len(binary_files)} binary files\n\n"
        )

        output.write("INCLUDED PATHS:\n")
        for path in INCLUDE_PATHS:
            output.write(f"  - {path.relative_to(BASE_DIR)}\n")

        output.write("\n")
        output.write("=" * 80 + "\n")
        output.write("FILES\n")
        output.write("=" * 80 + "\n\n")

        for file in text_files:
            relative_path = file.relative_to(BASE_DIR)

            output.write("\n")
            output.write("#" * 80 + "\n")
            output.write(f"# FILE: {relative_path}\n")
            output.write("#" * 80 + "\n\n")

            try:
                content = file.read_text(encoding="utf-8")
                output.write(content)
            except Exception as e:
                output.write(f"[ERROR READING FILE: {e}]\n")

            output.write("\n\n")

        if binary_files:
            output.write("\n")
            output.write("=" * 80 + "\n")
            output.write("BINARY FILES\n")
            output.write("=" * 80 + "\n\n")

            for file in binary_files:
                output.write(
                    f"- {file.relative_to(BASE_DIR)}\n"
                )

    print(f"Export created: {OUTPUT_FILE}")
    print(f"{len(text_files)} text files, {len(binary_files)} binary files")


if __name__ == "__main__":
    create_export()