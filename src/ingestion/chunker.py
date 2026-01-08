def chunk_file_by_separator(file_path, separator="---"):
    """
    Read a text file and split it into chunks based on a separator.

    Args:
        file_path (str or Path)
        separator (str): delimiter that separates logical chunks

    Returns:
        List[str]: list of chunk texts
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_chunks = [c.strip() for c in content.split(separator) if c.strip()]
    return raw_chunks


def chunk_documents_from_files(file_paths, separator="---"):
    """
    Take multiple files, split them into chunks by separator,
    and assign doc_id + chunk_id.

    Args:
        file_paths (List[str or Path])
        separator (str)

    Returns:
        List[dict]: each dict has 'doc_id', 'chunk_id', 'text'
    """
    all_chunks = []
    for file_index, path in enumerate(file_paths):
        chunks = chunk_file_by_separator(path, separator)
        for i, c in enumerate(chunks):
            all_chunks.append({
                "doc_id": f"doc_{file_index}",
                "chunk_id": i,
                "text": c
            })
    return all_chunks