from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    from loader import load_pdf
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    pdf_files = list((project_root / "data" / "raw").glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in data/raw/. Add one and re-run.")
    else:
        docs = load_pdf(str(pdf_files[0]))
        chunks = chunk_documents(docs)

        print(f"Split {len(docs)} pages into {len(chunks)} chunks")
        print("--- First chunk ---")
        print(chunks[0].page_content)
        print("--- Metadata ---")
        print(chunks[0].metadata)


# metadata matters later for citing which chunk an answer came from