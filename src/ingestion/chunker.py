from pathlib import Path
import re

YEAR_REGEX = re.compile(r"^\d{4}$")


def chunk_file_by_year(file_path):
    """
    Parse a document with:
      - Title
      - Type
      - Year-based sections

    Returns:
        List[dict]: chunks with title, type, year, text,
                    start_char, end_char
    """
    raw_text = Path(file_path).read_text(encoding="utf-8")
    lines = raw_text.splitlines()

    title = None
    doc_type = None

    chunks = []
    current_year = None
    buffer = []

    buffer_start_char = None
    running_char_index = 0  # tracks position in raw_text

    for line in lines:
        stripped = line.rstrip()
        line_len = len(line) + 1  # +1 for newline

        if stripped.startswith("Title:"):
            title = stripped.replace("Title:", "").strip()
            running_char_index += line_len
            continue

        if stripped.startswith("Type:"):
            doc_type = stripped.replace("Type:", "").strip()
            running_char_index += line_len
            continue

        # New year section
        if YEAR_REGEX.match(stripped):
            if current_year and buffer:
                chunk_text = " ".join(buffer).strip()
                end_char = buffer_start_char + len(chunk_text)

                chunks.append({
                    "title": title,
                    "type": doc_type,
                    "year": current_year,
                    "text": chunk_text,
                    "start_char": buffer_start_char,
                    "end_char": end_char,
                })

                buffer = []
                buffer_start_char = None

            current_year = int(stripped)
            running_char_index += line_len
            continue

        # Normal content
        if stripped:
            if buffer_start_char is None:
                buffer_start_char = running_char_index
            buffer.append(stripped)

        running_char_index += line_len

    # Final chunk
    if current_year and buffer:
        chunk_text = " ".join(buffer).strip()
        end_char = buffer_start_char + len(chunk_text)

        chunks.append({
            "title": title,
            "type": doc_type,
            "year": current_year,
            "text": chunk_text,
            "start_char": buffer_start_char,
            "end_char": end_char,
        })

    return chunks


def chunk_documents_from_files(file_paths):
    """
    Chunk all files into year-based chunks with character offsets.

    Returns:
        List[dict]
    """
    all_chunks = []

    for path in file_paths:
        doc_name = Path(path).name

        year_chunks = chunk_file_by_year(path)
        for chunk_id, c in enumerate(year_chunks):
            all_chunks.append({
                "doc_name": doc_name,
                "chunk_id": chunk_id,
                **c
            })

    return all_chunks