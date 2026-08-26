from pathlib import Path
from ragas.testset import TestsetGenerator

from src.ingestion.loader import load_pdf
from src.eval.ragas_eval import get_ragas_llm, get_ragas_embeddings

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    pdf_files = list((project_root / "data" / "raw").glob("*.pdf"))
    docs = load_pdf(str(pdf_files[0]))

    generator = TestsetGenerator(
        llm=get_ragas_llm(),
        embedding_model=get_ragas_embeddings(),
    )

    testset = generator.generate_with_langchain_docs(docs, testset_size=5)

    df = testset.to_pandas()
    print(df[["user_input", "reference"]].to_string())

    output_path = project_root / "data" / "processed" / "golden_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved golden dataset to {output_path}")