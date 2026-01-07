def chunk_text(text, chunk_size=300, overlap=50):
    """
    Break a document into chunks with optional overlap.

    Args:
        text (str): The raw text of the document.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words overlapping between chunks.

    Returns:
        List[str]: List of chunk texts.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  

    return chunks


def chunk_documents(documents, chunk_size=300, overlap=50):
    """
    Apply chunking to a list of documents.

    Args:
        documents (List[dict]): Each dict has 'id' and 'text'.
        chunk_size (int)
        overlap (int)

    Returns:
        List[dict]: Each dict has 'doc_id', 'chunk_id', 'text'
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc["id"],
                "chunk_id": i,
                "text": chunk
            })
    return all_chunks