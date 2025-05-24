import os
import re
from bs4 import BeautifulSoup
import markdown2

# Config
DOCS_PATH = "D:\\Masrafe\\Coding\\Git_Hub_code\\ml_project\\Api_Doc_Ai_agent\\Agent\\docs\\Stripe API Reference.html"  # Can be .md or .html

def load_document(file_path):
    ext = os.path.splitext(file_path)[1]
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if ext == ".html":
        return html_to_text(content)
    elif ext == ".md":
        return markdown_to_text(content)
    else:
        raise ValueError("Unsupported file type")

def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    return soup.get_text(separator="\n")

def markdown_to_text(md_content):
    html = markdown2.markdown(md_content)
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n")

def chunk_text(text, max_chars=1000, overlap=200):
    """Split text by section headers and paragraphs into overlapping chunks."""
    paragraphs = re.split(r'\n{2,}', text)  # split by double newlines
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chars:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            # Add overlap from end of last chunk
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

if __name__ == "__main__":
    raw_text = load_document(DOCS_PATH)
    chunks = chunk_text(raw_text)

    print(f"✅ Loaded and chunked {DOCS_PATH}")
    print(f"Generated {len(chunks)} chunks.\n")

    # Sample output
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk #{i+1} ---\n{chunk[:500]}")