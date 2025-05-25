from chunk_docs import chunk_docs
from embed_docs import embed_docs
from query_agent import answer_question

def main():
    print("Step 1: Chunking the document...")
    chunks = chunk_docs()  

    print("\nStep 2: Embedding and storing chunks...")
    embed_docs(chunks)     

    print("\nStep 3: Ready to answer questions.")
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() in {"exit", "quit"}:
            break
        try:
            answer, _ = answer_question(question)
            print("\nAnswer:\n", answer)
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
