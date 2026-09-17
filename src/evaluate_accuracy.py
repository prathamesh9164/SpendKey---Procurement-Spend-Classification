"""
evaluate_accuracy.py

Computes tier-by-tier classification accuracy, high-confidence precision,
and governance metrics against the verified 189-transaction Gold Standard dataset.
"""

import os
import sys
import pandas as pd

def run_accuracy_evaluation(
    predictions_path: str = "output/final_classification.xlsx",
    gold_standard_path: str = "data/gold_standard.csv",
    print_report: bool = True
) -> dict:
    if not os.path.exists(predictions_path):
        # Fallback to final_classification.xlsx if v2 not found
        predictions_path = "output/final_classification.xlsx"

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    if not os.path.exists(gold_standard_path):
        raise FileNotFoundError(f"Gold standard file not found: {gold_standard_path}")

    genai_df = pd.read_excel(predictions_path) if predictions_path.endswith(".xlsx") else pd.read_csv(predictions_path)
    gold_df = pd.read_csv(gold_standard_path)

    merged = genai_df.merge(
        gold_df[[
            "transaction_id",
            "final_ground_truth_L1",
            "final_ground_truth_L2",
            "final_ground_truth_L3",
            "final_ground_truth_L4",
            "final_full_path"
        ]],
        on="transaction_id",
        how="inner"
    )

    n_total = len(merged)

    # Tier-by-Tier Accuracy
    l1_corr = (merged["l1"].astype(str).str.strip().str.lower() == merged["final_ground_truth_L1"].astype(str).str.strip().str.lower())
    l2_corr = (merged["l2"].astype(str).str.strip().str.lower() == merged["final_ground_truth_L2"].astype(str).str.strip().str.lower())
    l3_corr = (merged["l3"].astype(str).str.strip().str.lower() == merged["final_ground_truth_L3"].astype(str).str.strip().str.lower())
    l4_corr = (merged["l4"].astype(str).str.strip().str.lower() == merged["final_ground_truth_L4"].astype(str).str.strip().str.lower())
    full_corr = l1_corr & l2_corr & l3_corr & l4_corr

    # Confidence Breakdown
    high_conf = merged[merged["confidence_level"] == "HIGH"]
    high_conf_corr = (high_conf["l4"].astype(str).str.strip().str.lower() == high_conf["final_ground_truth_L4"].astype(str).str.strip().str.lower())

    med_conf = merged[merged["confidence_level"] == "MEDIUM"]
    med_conf_corr = (med_conf["l4"].astype(str).str.strip().str.lower() == med_conf["final_ground_truth_L4"].astype(str).str.strip().str.lower())

    # Governance Routing
    accepted = merged[merged["final_status"] == "ACCEPTED"]
    accepted_corr = (accepted["l4"].astype(str).str.strip().str.lower() == accepted["final_ground_truth_L4"].astype(str).str.strip().str.lower())

    review = merged[merged["final_status"] == "REVIEW_REQUIRED"]
    review_corr = (review["l4"].astype(str).str.strip().str.lower() == review["final_ground_truth_L4"].astype(str).str.strip().str.lower())

    metrics = {
        "total_evaluated": n_total,
        "l1_accuracy_pct": round(float(l1_corr.mean() * 100), 2),
        "l1_correct_count": int(l1_corr.sum()),
        "l2_accuracy_pct": round(float(l2_corr.mean() * 100), 2),
        "l2_correct_count": int(l2_corr.sum()),
        "l3_accuracy_pct": round(float(l3_corr.mean() * 100), 2),
        "l3_correct_count": int(l3_corr.sum()),
        "l4_accuracy_pct": round(float(l4_corr.mean() * 100), 2),
        "l4_correct_count": int(l4_corr.sum()),
        "full_path_accuracy_pct": round(float(full_corr.mean() * 100), 2),
        "full_path_correct_count": int(full_corr.sum()),
        "auto_accepted_precision_pct": round(float(accepted_corr.mean() * 100), 2),
        "auto_accepted_count": len(accepted),
        "high_confidence_precision_pct": round(float(high_conf_corr.mean() * 100), 2),
        "high_confidence_count": len(high_conf),
        "review_required_count": len(review),
    }

    if print_report:
        print("=" * 65)
        print("SPENDKEY GENAI CLASSIFIER - ACCURACY EVALUATION REPORT")
        print("=" * 65)
        print(f"Total Transactions Evaluated: {metrics['total_evaluated']} (against Gold Standard)")
        print("-" * 65)
        print(f"  Level 1 (Segment) Accuracy:     {metrics['l1_accuracy_pct']}%  ({metrics['l1_correct_count']}/{metrics['total_evaluated']})")
        print(f"  Level 2 (Family) Accuracy:      {metrics['l2_accuracy_pct']}%  ({metrics['l2_correct_count']}/{metrics['total_evaluated']})")
        print(f"  Level 3 (Category) Accuracy:    {metrics['l3_accuracy_pct']}%  ({metrics['l3_correct_count']}/{metrics['total_evaluated']})")
        print(f"  Level 4 (Commodity) Accuracy:   {metrics['l4_accuracy_pct']}%  ({metrics['l4_correct_count']}/{metrics['total_evaluated']})")
        print(f"  Full 4-Level Path Accuracy:     {metrics['full_path_accuracy_pct']}%  ({metrics['full_path_correct_count']}/{metrics['total_evaluated']})")
        print("-" * 65)
        print("GOVERNANCE & TRUST METRICS:")
        print(f"  Auto-Accepted Precision:        {metrics['auto_accepted_precision_pct']}%  ({metrics['auto_accepted_count']} items auto-passed)")
        print(f"  High-Confidence Precision:      {metrics['high_confidence_precision_pct']}%  ({metrics['high_confidence_count']} items)")
        print(f"  Human Review Routing:           {metrics['review_required_count']} items properly flagged for human review")
        print("=" * 65)

    return metrics

if __name__ == "__main__":
    run_accuracy_evaluation()
