from pathlib import Path
import pandas as pd

STOCHASTIC_METRICS = ["faithfulness", "answer_relevancy"]
RETRIEVAL_METRICS = ["llm_context_precision_with_reference", "context_recall"]

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    run1 = pd.read_csv(processed_dir / "eval_run_1.csv")
    run2 = pd.read_csv(processed_dir / "eval_run_2.csv")
    run3 = pd.read_csv(processed_dir / "eval_run_3.csv")
    pr2 = pd.read_csv(processed_dir / "eval_run_pr_2.csv")
    pr3 = pd.read_csv(processed_dir / "eval_run_pr_3.csv")

    # Faithfulness + answer_relevancy: 3 samples (run1, run2, run3)
    stochastic_all = pd.concat([
        run1[["user_input"] + STOCHASTIC_METRICS],
        run2[["user_input"] + STOCHASTIC_METRICS],
        run3[["user_input"] + STOCHASTIC_METRICS],
    ], ignore_index=True)
    stochastic_grouped = stochastic_all.groupby("user_input")[STOCHASTIC_METRICS].agg(["mean", "std"])
    stochastic_grouped.columns = ["_".join(col) for col in stochastic_grouped.columns]
    stochastic_grouped = stochastic_grouped.reset_index()

    # Context precision + recall: 3 samples (run1, pr2, pr3)
    retrieval_all = pd.concat([
        run1[["user_input"] + RETRIEVAL_METRICS],
        pr2[["user_input"] + RETRIEVAL_METRICS],
        pr3[["user_input"] + RETRIEVAL_METRICS],
    ], ignore_index=True)
    retrieval_grouped = retrieval_all.groupby("user_input")[RETRIEVAL_METRICS].agg(["mean", "std"])
    retrieval_grouped.columns = ["_".join(col) for col in retrieval_grouped.columns]
    retrieval_grouped = retrieval_grouped.reset_index()

    baseline = stochastic_grouped.merge(retrieval_grouped, on="user_input", how="left")

    output_path = processed_dir / "eval_results_baseline.csv"
    baseline.to_csv(output_path, index=False)

    print(baseline.to_string())
    print(f"\nSaved fully variance-aware baseline (3 samples per metric) to {output_path}")