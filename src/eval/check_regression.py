from pathlib import Path
import pandas as pd

from src.eval.build_baseline import run_full

METRICS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
STD_MULTIPLIER = 1.5
MIN_TOLERANCE = 0.05  # floor for metrics with baseline_std == 0

def check_regressions(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    merged = new_df.merge(baseline_df, on="user_input", suffixes=("_new", ""))

    flags = []
    for _, row in merged.iterrows():
        for metric in METRICS:
            new_val = row[f"{metric}_new"]
            baseline_mean = row[f"{metric}_mean"]
            baseline_std = row[f"{metric}_std"]

            tolerance = max(STD_MULTIPLIER * baseline_std, MIN_TOLERANCE)
            threshold = baseline_mean - tolerance
            regressed = new_val < threshold

            flags.append({
                "user_input": row["user_input"][:60] + "...",
                "metric": metric,
                "baseline_mean": round(baseline_mean, 3),
                "tolerance": round(tolerance, 3),
                "new_value": round(new_val, 3),
                "regressed": regressed,
            })

    return pd.DataFrame(flags)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    golden_df = pd.read_csv(processed_dir / "golden_dataset.csv")
    baseline_df = pd.read_csv(processed_dir / "eval_results_baseline.csv")

    print("Running fresh eval against current pipeline...")
    new_df = run_full(golden_df, top_k=8)

    report = check_regressions(baseline_df, new_df)

    print("\n--- Regression Report ---")
    print(report.to_string())

    n_regressed = report["regressed"].sum()
    if n_regressed > 0:
        print(f"\n  {n_regressed} regression(s) detected:")
        print(report[report["regressed"]].to_string())
    else:
        print("\n No regressions detected — all metrics within expected variance.")