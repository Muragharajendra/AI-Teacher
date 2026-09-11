import re
from pathlib import Path


remove_keyword = {
    "New words", "New Word", "Keywords", "Glossary", "Key Terms",
    "Terms to Know", "Words to Know",

    "Activity", "Activities", "Discuss", "Discussion",
    "Questions", "Question", "Exercises", "Exercise",

    "Check Your Progress", "Let's Discuss", "Let us discuss",
    "Let's Explore", "Let us explore",
    "Let's Work", "Let us work",
    "Let's Learn", "Let us learn",
    "Let's Read", "Let us read",
    "Let's Write", "Let us write",
    "Let's Listen", "Let us listen",
    "Let's Speak", "Let us speak",

    "Think About It", "Think and Discuss", "Think and Answer",
    "Think About", "Think and Respond",

    "Try This", "Try These", "Try it Yourself",
    "Do It Yourself", "Do this", "Do These",

    "Work in Pairs", "Work in Groups",
    "Discuss in Pairs", "Discuss in Groups",
    "Pair Work", "Group Work",

    "Before You Read", "Before we read", "Before Reading",
    "While You Read", "While we read", "While Reading",
    "After You Read", "After we read", "After Reading",

    "Read and Find Out", "Read and Answer", "Read and Discuss",

    "Learning Objectives", "Learning Objective", "Objectives",
    "Learning Outcomes", "Learning Outcome",

    "Source", "Sources",
    "Source A", "Source B", "Source C", "Source D",
    "Source E", "Source F",

    "Box", "Box 1", "Box 2", "Box 3",
    "Box 4", "Box 5", "Box 6",

    "Case Study", "Case Studies",
    "In Focus", "Focus",
    "Did You Know", "Do You Know",

    "Image", "Images",
    "Figure", "Figures",
    "Fig.", "Fig",

    "Map", "Maps",
    "Chart", "Charts",
    "Diagram", "Diagrams",

    "Illustration", "Illustrations",
    "Photo", "Photos",
    "Photograph", "Photographs",
    "Picture", "Pictures",
    "Caption", "Captions",

    "Project", "Projects",
    "Assignment", "Assignments",
    "Project Work", "Research Project",
    "Group Project", "Individual Project",
    "Field Work", "Fieldwork",
    "Survey", "Activity Project",

    "Summary", "Summaries",
    "Review", "Reviews",
    "Revision", "Recap",
    "Key Points", "Key Point",
    "Points to Remember",
    "Remember", "Remember This",
    "Quick Review", "Quick Recap",
    "Chapter Summary", "Chapter Review", "Chapter Recap",

    "Practice", "Practice Questions", "Practice Question",
    "Test Yourself", "Test Your Knowledge",
    "Self Assessment", "Self-Assessment",
    "Assessment", "Assess Yourself",
    "Evaluate Yourself", "Evaluation",

    "Quiz", "Quizzes",
    "Multiple Choice Questions",
    "MCQ", "MCQs",

    "Short Answer Questions",
    "Long Answer Questions",
    "Very Short Answer Questions",

    "Writing", "Writing Activity", "Writing Activities",
    "Speaking", "Speaking Activity",
    "Listening", "Listening Activity",
    "Reading Activity", "Reading Activities",

    "Explore", "Explore More", "Explore Further",
    "Learn More", "Find Out", "Find Out More",
    "Find More",

    "References", "Reference",
    "Further Reading", "Further Readings",
    "Additional Reading", "Additional Readings",
    "Suggested Reading", "Suggested Readings",
    "Additional Resources", "Resources",
    "Bibliography",

    "Acknowledgements", "Acknowledgement",
    "Credits", "Credit",
    "Copyright", "Copyrights",

    "Preface", "Foreword",
    "About the Book", "About the Author",
    "About the Authors",

    "Contents", "Index",
    "Appendix", "Appendices",
    "Notes", "Note",
    "Endnotes", "Footnotes",

    "_Source_"
}


def clean_markdown(text: str) -> str:

    # 1. Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 2. Remove PyMuPDF4LLM picture-text blocks
    text = re.sub(
        r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 3. Remove <mark> but preserve content
    text = re.sub(
        r"<mark>(.*?)</mark>",
        r"\1",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # 4. Remove <br>
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # 5. Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # 6. Strip each line
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    # 7. Normalize blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_heading_text(text: str) -> str:

    text = re.sub(
        r"<mark>(.*?)</mark>",
        r"\1",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")

    return text.strip()


def normalize_headings(text: str) -> str:

    lines = text.splitlines()
    normalized = []

    for line in lines:

        match = re.match(
            r"^(#{1,6})\s+(.*)$",
            line
        )

        if not match:
            normalized.append(line)
            continue

        hashes = match.group(1)
        heading_text = match.group(2)

        heading_text = clean_heading_text(
            heading_text
        )

        normalized.append(
            f"{hashes} {heading_text}"
        )

    return "\n".join(normalized)


def normalize_heading_levels(text: str) -> str:

    lines = text.splitlines()
    normalized = []

    previous_level = 0

    for line in lines:

        match = re.match(
            r"^(#{1,6})\s+(.*)$",
            line
        )

        if not match:
            normalized.append(line)
            continue

        level = len(match.group(1))
        heading_text = match.group(2)

        if previous_level == 0:
            new_level = level

        elif level > previous_level + 1:
            new_level = previous_level + 1

        else:
            new_level = level

        new_level = min(new_level, 6)

        normalized.append(
            f"{'#' * new_level} {heading_text}"
        )

        previous_level = new_level

    return "\n".join(normalized)


def remove_headers(text):

    lines = text.splitlines()
    cleaned_lines = []

    keywords = {
        keyword.strip().lower()
        for keyword in remove_keyword
    }

    for line in lines:

        stripped = line.strip()

        # Detect Markdown heading
        match = re.match(
            r"^#{1,6}\s*(.*?)\s*$",
            stripped
        )

        if match:

            heading = match.group(1)

            # Remove markdown formatting
            heading = re.sub(
                r"[*_`]",
                "",
                heading
            )

            # Normalize spaces
            heading = re.sub(
                r"\s+",
                " ",
                heading
            ).strip().lower()

            # Exact match
            if heading in keywords:
                continue

            # Match things like:
            # Activity 1
            # Discuss 3f
            # Source A
            # Box 2
            for keyword in keywords:

                pattern = rf"^{re.escape(keyword)}(?:\s+\S+)*$"

                if re.fullmatch(pattern, heading):
                    break

            else:
                cleaned_lines.append(line)
                continue

            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_and_normalize_markdown(text: str) -> str:

    # First clean extraction artifacts
    text = clean_markdown(text)

    # Normalize heading formatting
    text = normalize_headings(text)

    # Normalize heading hierarchy
    text = normalize_heading_levels(text)

    # Remove unwanted headings
    text = remove_headers(text)

    return text.strip()

