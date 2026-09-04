from pathlib import Path
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig

from src.generation.pipeline import rag_query
from src.eval.ragas_eval import get_ragas_llm, get_ragas_embeddings

def run_full_eval(golden_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in golden_df.iterrows():
        question = row["user_input"]
        reference = row["reference"]

        answer, results = rag_query(question)
        contexts = [hit.payload["text"] for hit in results]

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
        })

    dataset = EvaluationDataset.from_list(rows)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=RunConfig(max_workers=1, timeout=300, max_retries=5),
    )
    return result.to_pandas()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    golden_df = pd.read_csv(project_root / "data" / "processed" / "golden_dataset.csv")
    df = run_full_eval(golden_df)
    print(df.to_string())