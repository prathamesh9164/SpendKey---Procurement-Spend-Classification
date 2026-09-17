"""
data_processing.py

Loads and cleans the two sheets we actually use from the Spendkey Excel workbook:

1. "Question Paper"            -> the 197 procurement transactions to classify
2. "Taxonomy Reference L1-L4"  -> the 256-row taxonomy (source of truth)

The workbook also contains a sheet called "_TaxonomyList", which is just a flat
list of the 256 "L1 > L2 > L3 > L4" strings used to power the Excel dropdown in
the Question Paper sheet. We use it only as a cross-check that our cleaned
taxonomy matches the dropdown exactly.

IMPORTANT: We never write back to the original Excel file. We only read it.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Column name constants (keeps the rest of the codebase readable)
# ---------------------------------------------------------------------------
COL_ID = "transaction_id"
COL_DESCRIPTION = "spend_description"
COL_VENDOR = "vendor"
COL_SOURCE_TYPE = "source_type"

TAX_L1, TAX_L2, TAX_L3, TAX_L4 = "L1", "L2", "L3", "L4"
TAX_FULL_PATH = "full_path"
TAX_BOUNDARY_NOTE = "boundary_note"


def load_workbook_info(excel_path: str) -> dict:
    """
    Inspect the workbook and return basic metadata (sheet names, shapes).
    Used for the "STEP 1: inspect the workbook" requirement of the assignment.
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Could not find the Excel file at: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    info = {"sheet_names": xls.sheet_names, "sheets": {}}

    for name in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=name, header=None)
        info["sheets"][name] = {"n_rows": raw.shape[0], "n_cols": raw.shape[1]}

    return info


def load_transactions(excel_path: str) -> pd.DataFrame:
    """
    Load the "Question Paper" sheet and return a clean DataFrame of
    procurement transactions with predictable column names.

    The real header row in this sheet is row 9 in Excel (index 8 in pandas),
    because rows 1-8 contain a title and instructions text.
    """
    raw = pd.read_excel(excel_path, sheet_name="Question Paper", header=8)

    df = raw.rename(
        columns={
            "#": COL_ID,
            "Spend Description": COL_DESCRIPTION,
            "Vendor / Supplier": COL_VENDOR,
            "Source Type": COL_SOURCE_TYPE,
        }
    )

    # Keep only the columns we actually need for classification.
    df = df[[COL_ID, COL_DESCRIPTION, COL_VENDOR, COL_SOURCE_TYPE]].copy()

    # Drop fully blank trailing rows, if any.
    df = df.dropna(subset=[COL_DESCRIPTION], how="all").reset_index(drop=True)

    # Basic text cleaning: strip whitespace, collapse repeated spaces.
    for col in [COL_DESCRIPTION, COL_VENDOR, COL_SOURCE_TYPE]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # If a description is genuinely missing/blank, mark it clearly instead of
    # silently sending an empty string to the LLM.
    df[COL_DESCRIPTION] = df[COL_DESCRIPTION].replace(
        {"": "MISSING_DESCRIPTION", "nan": "MISSING_DESCRIPTION"}
    )

    df[COL_ID] = df[COL_ID].astype(int)

    return df


def load_taxonomy(excel_path: str) -> pd.DataFrame:
    """
    Load the "Taxonomy Reference L1-L4" sheet and return a clean, fully
    populated DataFrame with one row per L4 commodity (the leaf level).

    The sheet uses merged cells for L1/L2/L3, so most rows only carry an L4
    value with the higher levels left blank. We forward-fill L1/L2/L3 so every
    row has the complete hierarchy.
    """
    raw = pd.read_excel(excel_path, sheet_name="Taxonomy Reference L1-L4", header=3)
    raw.columns = [TAX_L1, TAX_L2, TAX_L3, TAX_L4, TAX_FULL_PATH, TAX_BOUNDARY_NOTE]

    df = raw.dropna(how="all").copy()

    # Forward-fill the hierarchy levels that were merged in Excel.
    df[[TAX_L1, TAX_L2, TAX_L3]] = df[[TAX_L1, TAX_L2, TAX_L3]].ffill()

    # Clean text.
    for col in [TAX_L1, TAX_L2, TAX_L3, TAX_L4, TAX_FULL_PATH]:
        df[col] = df[col].astype(str).str.strip()

    df[TAX_BOUNDARY_NOTE] = df[TAX_BOUNDARY_NOTE].where(df[TAX_BOUNDARY_NOTE].notna(), "")
    df[TAX_BOUNDARY_NOTE] = df[TAX_BOUNDARY_NOTE].astype(str).str.strip()

    df = df.reset_index(drop=True)

    return df


def validate_taxonomy_against_dropdown_list(excel_path: str, taxonomy_df: pd.DataFrame) -> bool:
    """
    Cross-check that our cleaned taxonomy's full paths exactly match the
    "_TaxonomyList" sheet (the flat list that powers the Excel dropdown).
    Returns True if they match exactly.
    """
    dropdown = pd.read_excel(excel_path, sheet_name="_TaxonomyList", header=None)
    dropdown_set = set(dropdown[0].dropna().astype(str).str.strip().tolist())
    our_set = set(taxonomy_df[TAX_FULL_PATH].tolist())
    return dropdown_set == our_set


def get_data_quality_report(transactions_df: pd.DataFrame, taxonomy_df: pd.DataFrame) -> dict:
    """Return a small dictionary summarizing data quality checks."""
    return {
        "n_transactions": len(transactions_df),
        "n_taxonomy_leaf_nodes": len(taxonomy_df),
        "n_unique_l1_segments": taxonomy_df[TAX_L1].nunique(),
        "missing_descriptions": int(
            (transactions_df[COL_DESCRIPTION] == "MISSING_DESCRIPTION").sum()
        ),
        "duplicate_transaction_rows": int(
            transactions_df.duplicated(subset=[COL_DESCRIPTION, COL_VENDOR]).sum()
        ),
        "n_boundary_notes": int((taxonomy_df[TAX_BOUNDARY_NOTE] != "").sum()),
    }
