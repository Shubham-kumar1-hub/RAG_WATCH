from pathlib import Path
import pandas as pd

STOCHASTIC_METRICS = ["faithfulness", "answer_relevancy"]
FIXED_METRICS = ["llm_context_precision_with_reference", "context_recall"]

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    run1 = pd.read_csv(processed_dir / "eval_run_1.csv")
    run2 = pd.read_csv(processed_dir / "eval_run_2.csv")
    run3 = pd.read_csv(processed_dir / "eval_run_3.csv")

    stochastic_all = pd.concat([
        run1[["user_input"] + STOCHASTIC_METRICS],
        run2[["user_input"] + STOCHASTIC_METRICS],
        run3[["user_input"] + STOCHASTIC_METRICS],
    ], ignore_index=True)

    grouped = stochastic_all.groupby("user_input")[STOCHASTIC_METRICS].agg(["mean", "std"])
    grouped.columns = ["_".join(col) for col in grouped.columns]
    grouped = grouped.reset_index()

    fixed = run1[["user_input"] + FIXED_METRICS].rename(
        columns={m: f"{m}_baseline" for m in FIXED_METRICS}
    )

    baseline = grouped.merge(fixed, on="user_input", how="left")

    output_path = processed_dir / "eval_results_baseline.csv"
    baseline.to_csv(output_path, index=False)

    print(baseline.to_string())
    print(f"\nSaved final variance-aware baseline to {output_path}")