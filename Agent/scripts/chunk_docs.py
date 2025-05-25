import os
import re
from dotenv import load_dotenv
from load_docs import load_document

load_dotenv()
DOCS_PATH = os.getenv("DOCS_PATH")

def chunk_text(text, max_chars=1000, overlap=200):
    import re
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chars:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def chunk_docs():
    raw_text = load_document(DOCS_PATH)
    chunks = chunk_text(raw_text)
    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk #{i+1} ---\n{chunk[:500]}")
    return chunks

if __name__ == "__main__":
    chunk_docs()
