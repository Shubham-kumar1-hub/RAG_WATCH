from pathlib import Path
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig

from src.generation.pipeline import rag_query
from src.eval.ragas_eval import get_ragas_llm, get_ragas_embeddings

RUN_CONFIG = RunConfig(max_workers=1, timeout=300, max_retries=5)

def build_rows(golden_df, top_k=5):
    rows = []
    for _, row in golden_df.iterrows():
        answer, results = rag_query(row["user_input"], top_k=top_k)
        contexts = [hit.payload["text"] for hit in results]
        rows.append({
            "user_input": row["user_input"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": row["reference"],
        })
    return EvaluationDataset.from_list(rows)

def run_stochastic_only(golden_df):
    dataset = build_rows(golden_df)
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=RUN_CONFIG,
    )
    return result.to_pandas()

def run_full(golden_df, top_k=5):
    dataset = build_rows(golden_df, top_k=top_k)
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=RUN_CONFIG,
    )
    return result.to_pandas()

def run_precision_recall_only(golden_df):
    dataset = build_rows(golden_df)
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=RUN_CONFIG,
    )
    return result.to_pandas()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"
    golden_df = pd.read_csv(processed_dir / "golden_dataset.csv")

    # # Run 1: all 4 metrics — this fixes context_precision/recall for good
    # print("=== Run 1 (full) ===")
    # df1 = run_full(golden_df)
    # df1.to_csv(processed_dir / "eval_run_1.csv", index=False)
    # print(f"Saved run 1 to {processed_dir / 'eval_run_1.csv'}")

    # # Run 2: stochastic metrics only — cheaper, since precision/recall won't change
    # print("=== Run 2 (stochastic only) ===")
    # df2 = run_stochastic_only(golden_df)
    # df2.to_csv(processed_dir / "eval_run_2.csv", index=False)
    # print(f"Saved run 2 to {processed_dir / 'eval_run_2.csv'}")

    # print("=== Run 3 (stochastic only) ===")
    # df3 = run_stochastic_only(golden_df)
    # df3.to_csv(processed_dir / "eval_run_3.csv", index=False)
    # print(f"Saved run 3 to {processed_dir / 'eval_run_3.csv'}")

    # print("=== Precision/Recall Run 2 ===")
    # df = run_precision_recall_only(golden_df)
    # df.to_csv(processed_dir / "eval_run_pr_2.csv", index=False)
    # print(f"Saved to {processed_dir / 'eval_run_pr_2.csv'}")

    print("=== Precision/Recall Run 3 ===")
    df = run_precision_recall_only(golden_df)
    df.to_csv(processed_dir / "eval_run_pr_3.csv", index=False)
    print(f"Saved to {processed_dir / 'eval_run_pr_3.csv'}")