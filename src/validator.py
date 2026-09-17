"""
validator.py

We never trust the LLM's output blindly. This module checks every returned
L1/L2/L3/L4 combination against the real taxonomy DataFrame (the source of
truth) and decides whether the classification is VALID, INVALID, or needs
REVIEW, plus a final human-in-the-loop status.
"""

from src.data_processing import TAX_L1, TAX_L2, TAX_L3, TAX_L4, TAX_FULL_PATH


def detect_semantic_mismatches(transaction_text: str, l1: str, l2: str, l3: str, l4: str) -> tuple:
    """
    Check for domain-specific semantic contradictions between transaction text and predicted category.
    Returns (is_mismatch: bool, reason: str).
    """
    if not transaction_text or not l4:
        return False, ""

    txt = str(transaction_text).lower()
    l4_str = str(l4).lower()
    l1_str = str(l1).lower()

    # Rule 1: EPC / Energy Performance Survey vs Asbestos / Hazardous
    if ("epc" in txt or "energy performance" in txt) and ("asbestos" in l4_str or "waste" in l4_str):
        return True, "EPC energy compliance surveys cannot be classified as Asbestos management or hazardous waste."

    # Rule 2: PLC / Programmable Logic Controller vs Hydraulics / Fluid Power
    if ("plc" in txt or "programmable logic" in txt) and ("hydraulic" in l4_str or "pneumatic" in l4_str):
        return True, "Programmable logic controllers (PLCs) are electronic automation controls, not hydraulics/fluid power."

    # Rule 3: Uniforms / Workwear vs Paper / Office Stationery
    if ("uniform" in txt or "workwear" in txt or "scrubs" in txt) and ("paper" in l4_str or "stationery" in l4_str or "print" in l1_str):
        return True, "Staff workwear and uniforms belong in Workwear & PPE, never office stationery or print."

    # Rule 4: Clinical/Medical Locum Staffing vs Tax, Rates, Software, or Office supplies
    if ("locum" in txt or "surgical team" in txt or "medical cover" in txt) and (
        "tax" in l4_str or "rates" in l4_str or "software" in l4_str or "stationery" in l4_str or "finance" in l1_str
    ):
        return True, "Locum medical/clinical staffing cannot be classified as finance, taxes/rates, software, or office supplies."

    # Rule 5: Employer of Record (EOR) vs Direct Statutory Taxes
    if ("eor" in txt or "employer of record" in txt) and ("national insurance" in l4_str or "statutory payments" in l1_str):
        return True, "Employer of Record (EOR) is a third-party managed payroll service, not direct statutory NIC remittance."

    # Rule 6: Bundled Multi-Service Contracts (IFM / PFI) vs single service
    if ("ifm" in txt or "pfi unitary" in txt or "bundled contract" in txt):
        return True, "Bundled multi-service contract (IFM/PFI) encompasses multiple distinct services; requires line-item breakdown or human review."

    return False, ""


def validate_classification(result: dict, taxonomy_df) -> dict:
    """
    Check a single classification result dict against the taxonomy.

    Adds these keys to the result and returns it:
      - l1_valid, l2_valid, l3_valid, l4_valid (bool)
      - full_path_valid (bool)   -> the exact L1>L2>L3>L4 combination exists
      - semantic_mismatch (bool) -> domain inconsistency detected
      - semantic_mismatch_reason -> explanation of mismatch if detected
      - validation_status         -> "VALID" | "INVALID" | "REVIEW"
      - validation_notes          -> short human-readable explanation
    """
    api_err = result.get("api_error")
    has_api_error = api_err is not None and str(api_err).strip() not in ("", "nan", "None")
    if has_api_error:
        result["l1_valid"] = result["l2_valid"] = result["l3_valid"] = result["l4_valid"] = False
        result["full_path_valid"] = False
        result["semantic_mismatch"] = False
        result["semantic_mismatch_reason"] = None
        result["validation_status"] = "INVALID"
        result["validation_notes"] = f"No classification produced: {result['api_error']}"
        return result

    l1, l2, l3, l4 = result.get("l1"), result.get("l2"), result.get("l3"), result.get("l4")

    l1_valid = l1 in set(taxonomy_df[TAX_L1])
    l2_valid = l2 in set(taxonomy_df.loc[taxonomy_df[TAX_L1] == l1, TAX_L2])
    l3_valid = l3 in set(
        taxonomy_df.loc[(taxonomy_df[TAX_L1] == l1) & (taxonomy_df[TAX_L2] == l2), TAX_L3]
    )
    l4_valid = l4 in set(
        taxonomy_df.loc[
            (taxonomy_df[TAX_L1] == l1) & (taxonomy_df[TAX_L2] == l2) & (taxonomy_df[TAX_L3] == l3),
            TAX_L4,
        ]
    )
    full_path_valid = l1_valid and l2_valid and l3_valid and l4_valid

    result["l1_valid"] = l1_valid
    result["l2_valid"] = l2_valid
    result["l3_valid"] = l3_valid
    result["l4_valid"] = l4_valid
    result["full_path_valid"] = full_path_valid

    # Semantic mismatch validation
    desc = result.get("spend_description") or ""
    vendor = result.get("vendor") or ""
    tx_text = f"{desc} {vendor}".strip()

    is_mismatch, mismatch_reason = detect_semantic_mismatches(tx_text, l1, l2, l3, l4)
    result["semantic_mismatch"] = is_mismatch
    result["semantic_mismatch_reason"] = mismatch_reason if is_mismatch else None

    if not full_path_valid:
        result["validation_status"] = "INVALID"
        result["validation_notes"] = (
            "The taxonomy path returned by the LLM does not exist in the "
            "source-of-truth taxonomy (a level was invented, mismatched, or "
            "the levels don't actually chain together)."
        )
    elif is_mismatch:
        result["validation_status"] = "REVIEW"
        result["validation_notes"] = f"Semantic mismatch detected: {mismatch_reason}"
    elif result.get("human_review_required"):
        result["validation_status"] = "REVIEW"
        result["validation_notes"] = "LLM flagged this classification as ambiguous."
    else:
        result["validation_status"] = "VALID"
        result["validation_notes"] = "Taxonomy path confirmed to exist and chain correctly."

    return result


def apply_human_in_the_loop_rules(result: dict) -> dict:
    """
    Decide final_status from confidence_level + validation_status, per the
    assignment's stated rules:

      HIGH confidence + valid taxonomy         -> ACCEPTED
      MEDIUM confidence                        -> REVIEW_REQUIRED
      LOW confidence                           -> REVIEW_REQUIRED
      Invalid taxonomy path                    -> REVIEW_REQUIRED (see note)
      Ambiguous classification (flagged by LLM)-> REVIEW_REQUIRED
    """
    status = result.get("validation_status")
    confidence = result.get("confidence_level")

    if status == "INVALID":
        # An invented/mismatched taxonomy path is never auto-accepted, and we
        # keep it visibly separate from "just needs a second pair of eyes".
        result["final_status"] = "INVALID"
    elif status == "REVIEW":
        result["final_status"] = "REVIEW_REQUIRED"
    elif confidence == "HIGH":
        result["final_status"] = "ACCEPTED"
    elif confidence in ("MEDIUM", "LOW"):
        result["final_status"] = "REVIEW_REQUIRED"
    else:
        # Missing/unexpected confidence value -- do not guess, send for review.
        result["final_status"] = "REVIEW_REQUIRED"

    result["human_review_required"] = result["final_status"] in ("REVIEW_REQUIRED", "INVALID")

    return result


def validate_and_finalize(results: list, taxonomy_df) -> list:
    """Run validate_classification + apply_human_in_the_loop_rules over a
    list of classification result dicts."""
    finalized = []
    for result in results:
        result = validate_classification(result, taxonomy_df)
        result = apply_human_in_the_loop_rules(result)
        finalized.append(result)
    return finalized


def summarize_results(results: list) -> dict:
    """Compute the summary metrics required by the assignment, from the
    actual finalized results (never fabricated)."""
    total = len(results)
    accepted = sum(1 for r in results if r["final_status"] == "ACCEPTED")
    review = sum(1 for r in results if r["final_status"] == "REVIEW_REQUIRED")
    invalid = sum(1 for r in results if r["final_status"] == "INVALID")
    api_errors = sum(
        1 for r in results
        if r.get("api_error") is not None and str(r.get("api_error")).strip() not in ("", "nan", "None")
    )
    successfully_classified = total - api_errors

    return {
        "total_transactions": total,
        "successfully_classified": successfully_classified,
        "accepted": accepted,
        "review_required": review,
        "invalid": invalid,
        "api_errors": api_errors,
        "pct_accepted": round(100 * accepted / total, 1) if total else 0.0,
        "pct_review_required": round(100 * review / total, 1) if total else 0.0,
    }
