from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    pdf_files = list(raw_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the 'data/raw' directory. Add one and re-run.")
    else:
        docs = load_pdf((str(pdf_files[0])))
        print(f"Loaded {len(docs)} pages from {pdf_files[0].name}.")
        print("--- First page preview ---")
        print(docs[0].page_content[:300])


# docs[0] is page 1, docs[1] is page 2, etc.