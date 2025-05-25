import os
import markdown2
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
DOCS_PATH = os.getenv("DOCS_PATH")

def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    return soup.get_text(separator="\n")

def markdown_to_text(md_content):
    html = markdown2.markdown(md_content)
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n")

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

def load_docs():
    return load_document(DOCS_PATH)

if __name__ == "__main__":
    text = load_docs()
    print("Document loaded successfully.")
    print(text[:1000])
