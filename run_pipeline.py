import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from src.data_processing import load_transactions, load_taxonomy
from src.genai_classifier import get_groq_client, classify_all_transactions, generate_classification_quality_report
from src.validator import validate_and_finalize, summarize_results

def main():
    excel_path = "data/Spendkey_Assignment.xlsx"
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        sys.exit(1)

    print("=" * 60)
    print("STARTING FULL GENAI CLASSIFICATION PIPELINE")
    print("=" * 60)
    start_time = time.time()

    transactions_df = load_transactions(excel_path)
    taxonomy_df = load_taxonomy(excel_path)
    client = get_groq_client()

    # 1. Run batch classification with high-reasoning model
    all_results = classify_all_transactions(
        client=client,
        transactions_df=transactions_df,
        taxonomy_df=taxonomy_df,
        model="openai/gpt-oss-120b",
        batch_size=10,
        max_workers=1,
        request_interval=25.0,
        top_n_taxonomy=7
    )

    # 2. Save raw classifications
    results_df = pd.DataFrame(all_results)
    os.makedirs("output", exist_ok=True)
    results_df.to_csv("output/raw_classification_results.csv", index=False)
    print(f"Saved output/raw_classification_results.csv with {len(results_df)} records.")

    # 3. Validation & Human-in-the-Loop Governance
    raw_results = results_df.to_dict("records")
    finalized_results = validate_and_finalize(raw_results, taxonomy_df)
    finalized_df = pd.DataFrame(finalized_results)

    # 4. Merge with transaction metadata and save final Excel
    final_df = transactions_df.merge(
        finalized_df, on="transaction_id", how="left", suffixes=("", "_val")
    )

    output_columns = [
        "transaction_id", "spend_description", "vendor", "source_type",
        "l1", "l2", "l3", "l4",
        "classification_reason", "confidence_level",
        "validation_status", "human_review_required", "final_status",
        "retrieved_taxonomy"
    ]

    cols_to_use = [c for c in output_columns if c in final_df.columns]
    final_df = final_df[cols_to_use]

    output_file = "output/final_classification.xlsx"
    try:
        final_df.to_excel(output_file, index=False)
        print(f"Successfully saved final classification deliverable: {output_file} ({len(final_df)} records).")
    except PermissionError:
        alt_path = "output/final_classification_v2.xlsx"
        final_df.to_excel(alt_path, index=False)
        print(f"Notice: {output_file} was open/locked in another program. Saved to {alt_path} ({len(final_df)} records).")

    # 5. Quality & Summary Report
    total_time = time.time() - start_time
    print(f"\nPipeline finished in {total_time:.1f} seconds.")

    quality_report = generate_classification_quality_report(finalized_results)
    print("\n" + quality_report)

    summary = summarize_results(finalized_results)
    print("\nValidation Summary Metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
