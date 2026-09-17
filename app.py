"""
app.py

Simple Streamlit UI around the same pipeline used in the notebooks:
  upload Excel -> classify -> validate -> review table -> download Excel.

Run with:  streamlit run app.py
"""

import os
import io

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data_processing import load_transactions, load_taxonomy
from src.genai_classifier import get_groq_client, classify_all_transactions, GroqNotConfiguredError
from src.validator import validate_and_finalize, summarize_results

load_dotenv()

st.set_page_config(page_title="Spendkey GenAI Spend Classifier", layout="wide")
st.title("Spendkey GenAI Spend Classifier")
st.caption(
    "Upload the assessment workbook, classify every transaction against the "
    "L1-L4 taxonomy using Groq, and download the reviewed result."
)

uploaded_file = st.file_uploader("Upload Spendkey_Assignment.xlsx", type=["xlsx"])

if uploaded_file is not None:
    # The loader functions expect a path, so persist the upload temporarily.
    temp_path = "uploaded_workbook.xlsx"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    transactions_df = load_transactions(temp_path)
    taxonomy_df = load_taxonomy(temp_path)

    st.success(f"Loaded {len(transactions_df)} transactions and {len(taxonomy_df)} taxonomy rows.")
    st.dataframe(transactions_df.head(10), use_container_width=True)

    if st.button("Classify Transactions", type="primary"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error(
                "GROQ_API_KEY is not set. Add it to your .env file and restart "
                "the app. See .env.example."
            )
        else:
            client = get_groq_client()

            progress_bar = st.progress(0, text="Starting classification...")
            results = []

            # We re-implement the batch loop here (instead of calling
            # classify_all_transactions directly) so we can update a
            # progress bar row-by-row in the UI.
            cache = {}
            for i, (_, row) in enumerate(transactions_df.iterrows()):
                cache_key = (row["spend_description"], row["vendor"])
                if cache_key in cache:
                    result = dict(cache[cache_key])
                else:
                    from src.genai_classifier import classify_transaction
                    result = classify_transaction(client, row, taxonomy_df)
                    cache[cache_key] = result
                result["transaction_id"] = row["transaction_id"]
                results.append(result)

                progress_bar.progress(
                    (i + 1) / len(transactions_df),
                    text=f"Classified {i + 1}/{len(transactions_df)} transactions",
                )

            results = validate_and_finalize(results, taxonomy_df)
            summary = summarize_results(results)

            results_df = pd.DataFrame(results)
            final_df = transactions_df.merge(results_df, on="transaction_id", how="left")

            st.subheader("Summary Metrics")
            cols = st.columns(4)
            cols[0].metric("Total Transactions", summary["total_transactions"])
            cols[1].metric("Accepted", summary["accepted"])
            cols[2].metric("Review Required", summary["review_required"])
            cols[3].metric("Invalid", summary["invalid"])

            st.subheader("All Results")
            st.dataframe(final_df, use_container_width=True)

            st.subheader("Transactions Requiring Review")
            review_df = final_df[final_df["final_status"] != "ACCEPTED"]
            st.dataframe(review_df, use_container_width=True)

            buffer = io.BytesIO()
            final_df.to_excel(buffer, index=False, sheet_name="Final Classification")
            st.download_button(
                "Download final_classification.xlsx",
                data=buffer.getvalue(),
                file_name="final_classification.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
