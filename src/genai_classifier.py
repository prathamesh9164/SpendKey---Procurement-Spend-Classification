"""
Fast GenAI classifier for the SpendKey assignment.

Main optimization:
- Multiple transactions are classified in one Groq API request.
- Duplicate transactions are cached.
- Controlled parallel batch processing is used.
- Structured JSON output is used.
- Existing validator.py can validate the results afterward.
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from prompts.prompts import (
    SYSTEM_PROMPT,
    CLASSIFICATION_JSON_SCHEMA,
    BATCH_CLASSIFICATION_JSON_SCHEMA,
    build_classification_prompt,
    build_batch_classification_prompt,
    retrieve_relevant_taxonomy,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = os.environ.get(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

MAX_RETRIES = 3

RETRY_BACKOFF_SECONDS = 1.0

_CACHE_LOCK = threading.Lock()


# ============================================================
# CUSTOM ERROR
# ============================================================

class GroqNotConfiguredError(Exception):
    """Raised when GROQ_API_KEY is missing."""
    pass


# ============================================================
# CACHE
# ============================================================

def _get_disk_cache_path():
    """
    Location of the classification cache.

    The cache is stored inside:

    output/.classification_cache.json
    """

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # Go from src/ to project root
    project_dir = os.path.dirname(base_dir)

    output_dir = os.path.join(
        project_dir,
        "output"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    return os.path.join(
        output_dir,
        ".classification_cache.json"
    )


def load_disk_cache():
    """
    Load previously classified transactions.
    """

    cache_path = _get_disk_cache_path()

    if not os.path.exists(cache_path):
        return {}

    try:

        with open(
            cache_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


CACHE_VERSION = "v2"

def _cache_key(row):
    """
    Create a unique cache key using:
    CACHE_VERSION + Spend Description + Vendor
    """
    description = str(
        row.get(
            "spend_description",
            ""
        )
    ).strip()

    vendor = str(
        row.get(
            "vendor",
            ""
        )
    ).strip()

    return (
        f"{CACHE_VERSION}__||__{description}__||__{vendor}"
    )


def save_disk_cache_entry(
    cache_key,
    data
):
    """
    Save a classification result to disk.
    """

    if not data:
        return

    # Do not cache failed API calls
    if data.get("api_error"):
        return

    if not data.get("l1"):
        return

    cache_path = _get_disk_cache_path()

    with _CACHE_LOCK:

        try:

            cache = load_disk_cache()

            # transaction_id is not part of cache identity
            clean_data = {
                key: value
                for key, value in data.items()
                if key != "transaction_id"
            }

            cache[cache_key] = clean_data

            with open(
                cache_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    cache,
                    f,
                    indent=2
                )

        except Exception:
            pass


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    """
    Create and return Groq client.
    """

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise GroqNotConfiguredError(
            "GROQ_API_KEY is not set. "
            "Add it to the .env file."
        )

    from groq import Groq

    return Groq(
        api_key=api_key
    )


# ============================================================
# GROQ API CALL
# ============================================================

def _call_groq(
    client,
    model,
    prompt,
    response_schema
):
    """
    Send one request to Groq.

    Includes retry handling for temporary
    API failures and rate limits.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = client.chat.completions.create(

                model=model,

                temperature=0,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                response_format={
                    "type": "json_schema",
                    "json_schema": response_schema
                }
            )

            return (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as exc:

            last_error = exc

            error_text = str(
                exc
            ).lower()

            # --------------------------------------------
            # Try to detect explicit retry time (e.g. 10.5s or 9m10.8s)
            # --------------------------------------------

            match = re.search(
                r"try again in (?:(\d+)m)?(\d+\.?\d*)s",
                str(exc)
            )

            if match:
                mins = float(match.group(1) or 0)
                secs = float(match.group(2) or 0)
                wait_time = (mins * 60) + secs + 0.5

            elif (
                "429" in error_text
                or "rate" in error_text
            ):

                wait_time = max(
                    2.0,
                    RETRY_BACKOFF_SECONDS
                    * (2 ** attempt)
                )

            else:

                wait_time = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

            # --------------------------------------------
            # Final attempt or excessive wait time (>10s)
            # --------------------------------------------
            if wait_time > 10.0:
                print(
                    f"Groq rate limit required wait of {wait_time:.1f}s exceeds threshold (>10s). "
                    "Failing fast to protect daily quota and execution flow."
                )
                raise RuntimeError(
                    f"Groq API rate limit wait time ({wait_time:.1f}s) exceeded safe threshold (10s): {exc}"
                )

            if attempt == MAX_RETRIES:
                break

            print(
                f"Groq request failed ({type(exc).__name__}). "
                f"Retrying in {wait_time:.1f}s..."
            )

            time.sleep(
                wait_time
            )

    if last_error is not None:
        raise last_error


# ============================================================
# SINGLE TRANSACTION CLASSIFICATION
# ============================================================

def classify_transaction(
    client,
    transaction_row,
    taxonomy_df,
    model=DEFAULT_MODEL,
    top_n_taxonomy=7,
    use_disk_cache=True
):
    """
    Classify one transaction.

    This function is retained for:
    - testing
    - debugging
    - individual classification
    """

    cache_key = _cache_key(
        transaction_row
    )

    # --------------------------------------------
    # Check cache
    # --------------------------------------------

    if use_disk_cache:

        cache = load_disk_cache()

        if (
            cache_key in cache
            and cache[cache_key].get("l1")
        ):

            result = dict(
                cache[cache_key]
            )

            result["from_cache"] = True

            return result

    try:

        # ----------------------------------------
        # Retrieve relevant taxonomy
        # ----------------------------------------

        taxonomy_subset = (
            retrieve_relevant_taxonomy(
                (
                    f"{transaction_row['spend_description']} "
                    f"{transaction_row['vendor']}"
                ),
                taxonomy_df,
                top_n=top_n_taxonomy
            )
        )

        # ----------------------------------------
        # Build prompt
        # ----------------------------------------

        prompt = (
            build_classification_prompt(
                transaction_row,
                taxonomy_subset
            )
        )

        # ----------------------------------------
        # Call Groq
        # ----------------------------------------

        raw_response = _call_groq(
            client,
            model,
            prompt,
            CLASSIFICATION_JSON_SCHEMA
        )

        # ----------------------------------------
        # Parse JSON
        # ----------------------------------------

        result = json.loads(
            raw_response
        )

        # Single classification taxonomy path
        if result.get("l1") and result.get("l4"):
            result["retrieved_taxonomy"] = (
                f"{result['l1']} > {result['l2']} > {result['l3']} > {result['l4']}"
            )
        elif len(taxonomy_subset) > 0:
            result["retrieved_taxonomy"] = taxonomy_subset["full_path"].iloc[0]
        else:
            result["retrieved_taxonomy"] = None

        result["api_error"] = None

        result["from_cache"] = False

        # ----------------------------------------
        # Save result
        # ----------------------------------------

        if use_disk_cache:

            save_disk_cache_entry(
                cache_key,
                result
            )

        return result

    except Exception as exc:

        return {

            "l1": None,

            "l2": None,

            "l3": None,

            "l4": None,

            "classification_reason": None,

            "confidence_level": None,

            "human_review_required": True,

            "retrieved_taxonomy": None,

            "api_error": (
                f"{type(exc).__name__}: {exc}"
            ),

            "from_cache": False
        }


# ============================================================
# BUILD BATCH CONTEXT
# ============================================================

def _build_batch_context(
    rows,
    taxonomy_df,
    top_n_taxonomy=7
):
    """
    Retrieve a tailored taxonomy subset with boundary notes
    for every transaction in a batch.
    """

    contexts = []

    for row in rows:

        transaction_text = (
            f"{row['spend_description']} "
            f"{row['vendor']}"
        )

        taxonomy_subset = (
            retrieve_relevant_taxonomy(
                transaction_text,
                taxonomy_df,
                top_n=top_n_taxonomy
            )
        )

        paths = []
        for _, tax_row in taxonomy_subset.iterrows():
            full_p = tax_row["full_path"]
            note = tax_row["boundary_note"]
            if note and str(note).strip():
                paths.append(f"{full_p}  [Boundary note: {str(note).strip()}]")
            else:
                paths.append(full_p)

        contexts.append({

            "transaction_id": int(
                row["transaction_id"]
            ),

            "paths": paths,

            "raw_paths": (
                taxonomy_subset[
                    "full_path"
                ]
                .tolist()
            ),
        })

    return contexts


# ============================================================
# BATCH CLASSIFICATION
# ============================================================

def classify_batch(
    client,
    rows,
    taxonomy_df,
    model=DEFAULT_MODEL,
    top_n_taxonomy=7
):
    """
    Classify multiple transactions in ONE
    Groq API request.

    Example:

    10 transactions
        ↓
    1 Groq request
        ↓
    10 classifications
    """

    # --------------------------------------------
    # Build taxonomy context
    # --------------------------------------------

    contexts = _build_batch_context(
        rows,
        taxonomy_df,
        top_n_taxonomy
    )

    # --------------------------------------------
    # Build batch prompt
    # --------------------------------------------

    prompt = (
        build_batch_classification_prompt(
            rows,
            contexts
        )
    )

    # --------------------------------------------
    # Call Groq once
    # --------------------------------------------

    raw_response = _call_groq(
        client,
        model,
        prompt,
        BATCH_CLASSIFICATION_JSON_SCHEMA
    )

    # --------------------------------------------
    # Parse JSON
    # --------------------------------------------

    parsed = json.loads(
        raw_response
    )

    if isinstance(parsed, dict) and "classifications" in parsed:
        results = parsed["classifications"]
    elif isinstance(parsed, list):
        results = parsed
    else:
        raise ValueError(
            "Batch LLM response did not contain a valid list of classifications."
        )

    # --------------------------------------------
    # Convert list to dictionary
    # --------------------------------------------
    #
    # Key:
    # transaction_id
    #
    # This means response order does not matter.
    #

    top_candidate_map = {
        ctx["transaction_id"]: (ctx["raw_paths"][0] if ctx.get("raw_paths") else None)
        for ctx in contexts
    }

    result_dict = {}

    for item in results:

        transaction_id = int(
            item["transaction_id"]
        )

        # Single chosen taxonomy path
        if item.get("l1") and item.get("l4"):
            item["retrieved_taxonomy"] = (
                f"{item['l1']} > {item['l2']} > {item['l3']} > {item['l4']}"
            )
        else:
            item["retrieved_taxonomy"] = top_candidate_map.get(
                transaction_id,
                None
            )

        result_dict[
            transaction_id
        ] = item

    return result_dict


# ============================================================
# FAST CLASSIFICATION OF ALL TRANSACTIONS
# ============================================================

def classify_all_transactions(
    client,
    transactions_df,
    taxonomy_df,
    model=DEFAULT_MODEL,
    batch_size=10,
    max_workers=3,
    request_interval=0.2,
    top_n_taxonomy=5
):
    """
    Fast batch classification.

    Instead of:

        1 transaction -> 1 API call

    we use:

        10 transactions -> 1 API call

    For approximately 197 transactions:

        197 API calls
              ↓
        approximately 20 API calls
    """

    # ========================================================
    # 1. LOAD CACHE
    # ========================================================

    cache = load_disk_cache()

    # ========================================================
    # 2. REMOVE DUPLICATE TRANSACTIONS
    # ========================================================

    unique_rows = []

    seen = set()

    for _, row in transactions_df.iterrows():

        key = _cache_key(
            row
        )

        if key not in seen:

            seen.add(
                key
            )

            unique_rows.append(
                (
                    key,
                    row
                )
            )

    # ========================================================
    # 3. FIND UNCACHED TRANSACTIONS
    # ========================================================

    uncached = []

    for key, row in unique_rows:

        if (
            key not in cache
            or not cache[key].get("l1")
        ):

            uncached.append(
                (
                    key,
                    row
                )
            )

    cached_count = (
        len(unique_rows)
        - len(uncached)
    )

    print()
    print("=" * 60)
    print("FAST GENAI CLASSIFICATION")
    print("=" * 60)

    print(
        f"Total transactions: "
        f"{len(transactions_df)}"
    )

    print(
        f"Unique transactions: "
        f"{len(unique_rows)}"
    )

    print(
        f"Already cached: "
        f"{cached_count}"
    )

    print(
        f"Need Groq classification: "
        f"{len(uncached)}"
    )

    # ========================================================
    # 4. IF EVERYTHING IS CACHED
    # ========================================================

    if not uncached:

        print()
        print(
            "All transactions are already "
            "available in cache."
        )

        final_results = []

        for _, row in (
            transactions_df.iterrows()
        ):

            key = _cache_key(
                row
            )

            result = dict(
                cache[key]
            )

            result["transaction_id"] = int(
                row["transaction_id"]
            )

            result["from_cache"] = True

            final_results.append(
                result
            )

        return final_results

    # ========================================================
    # 5. CREATE BATCHES
    # ========================================================

    batches = []

    for i in range(
        0,
        len(uncached),
        batch_size
    ):

        batch = uncached[
            i:i + batch_size
        ]

        batches.append(
            batch
        )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Groq API requests required: "
        f"{len(batches)}"
    )

    print(
        f"Parallel workers: "
        f"{max_workers}"
    )

    print("=" * 60)
    print()

    # ========================================================
    # 6. PROCESS ONE BATCH
    # ========================================================

    def process_batch(
        batch_number,
        batch
    ):
        """
        Process one batch.
        """

        # Small stagger between workers.
        # This helps reduce rate-limit problems.
        if request_interval > 0:

            time.sleep(
                request_interval
                * batch_number
                / max(
                    1,
                    max_workers
                )
            )

        rows = [
            row
            for _, row
            in batch
        ]

        try:

            results = classify_batch(
                client,
                rows,
                taxonomy_df,
                model=model,
                top_n_taxonomy=top_n_taxonomy
            )

            # --------------------------------------------
            # Save every successful classification
            # --------------------------------------------

            for key, row in batch:

                transaction_id = int(
                    row["transaction_id"]
                )

                result = results.get(
                    transaction_id
                )

                if result is not None:

                    result["api_error"] = None

                    result["from_cache"] = False

                    save_disk_cache_entry(
                        key,
                        result
                    )

            return results

        except Exception as exc:

            print()
            print(
                f"Batch {batch_number + 1} "
                f"failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            # --------------------------------------------
            # Return review-required results
            # --------------------------------------------

            failed_results = {}

            for _, row in batch:

                transaction_id = int(
                    row["transaction_id"]
                )

                failed_results[
                    transaction_id
                ] = {

                    "l1": None,

                    "l2": None,

                    "l3": None,

                    "l4": None,

                    "classification_reason": None,

                    "confidence_level": None,

                    "human_review_required": True,

                    "retrieved_taxonomy": None,

                    "api_error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),

                    "from_cache": False
                }

            return failed_results

    # ========================================================
    # 7. RUN BATCHES IN PARALLEL
    # ========================================================

    all_new_results = {}

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {}

        for batch_number, batch in enumerate(
            batches
        ):

            future = executor.submit(
                process_batch,
                batch_number,
                batch
            )

            futures[future] = batch_number

        # --------------------------------------------
        # Progress bar
        # --------------------------------------------

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Classifying batches",
            unit="batch"
        ):

            try:

                results = future.result()

                all_new_results.update(
                    results
                )

            except Exception as exc:

                print(
                    f"Unexpected batch error: "
                    f"{exc}"
                )

    # ========================================================
    # 8. UPDATE CACHE IN MEMORY
    # ========================================================

    updated_cache = load_disk_cache()

    # ========================================================
    # 9. REBUILD RESULTS IN ORIGINAL ORDER
    # ========================================================

    final_results = []

    for _, row in (
        transactions_df.iterrows()
    ):

        key = _cache_key(
            row
        )

        transaction_id = int(
            row["transaction_id"]
        )

        # --------------------------------------------
        # Prefer newly generated result
        # --------------------------------------------

        if transaction_id in all_new_results:

            result = dict(
                all_new_results[
                    transaction_id
                ]
            )

            result["from_cache"] = False

        # --------------------------------------------
        # Otherwise use cache
        # --------------------------------------------

        elif key in updated_cache:

            result = dict(
                updated_cache[
                    key
                ]
            )

            result["from_cache"] = True

        # --------------------------------------------
        # No result
        # --------------------------------------------

        else:

            result = {

                "l1": None,

                "l2": None,

                "l3": None,

                "l4": None,

                "classification_reason": None,

                "confidence_level": None,

                "human_review_required": True,

                "retrieved_taxonomy": None,

                "api_error": (
                    "No classification result returned."
                ),

                "from_cache": False
            }

        result[
            "transaction_id"
        ] = transaction_id

        final_results.append(
            result
        )

    # ========================================================
    # 10. SUMMARY
    # ========================================================

    successful = sum(
        1
        for result in final_results
        if result.get("l1")
    )

    failed = (
        len(final_results)
        - successful
    )

    print()
    print("=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)

    print(
        f"Total results: "
        f"{len(final_results)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Need review / failed: "
        f"{failed}"
    )

    print("=" * 60)

    return final_results


# ============================================================
# CLASSIFICATION QUALITY REPORT
# ============================================================

def generate_classification_quality_report(results: list) -> str:
    """
    Generate an executive summary report of classification quality,
    concurrency, caching, and confidence distribution.
    """
    total = len(results)
    if total == 0:
        return "No classification results to report."

    from_cache_count = sum(1 for r in results if r.get("from_cache"))
    new_classified = total - from_cache_count

    high_conf = sum(1 for r in results if str(r.get("confidence_level")).upper() == "HIGH")
    med_conf = sum(1 for r in results if str(r.get("confidence_level")).upper() == "MEDIUM")
    low_conf = sum(1 for r in results if str(r.get("confidence_level")).upper() == "LOW")

    accepted = sum(1 for r in results if r.get("final_status") == "ACCEPTED")
    review_req = sum(1 for r in results if r.get("final_status") == "REVIEW_REQUIRED" or r.get("human_review_required"))
    invalid = sum(1 for r in results if r.get("final_status") == "INVALID" or r.get("api_error"))
    mismatches = sum(1 for r in results if r.get("semantic_mismatch"))

    lines = [
        "=" * 60,
        "SPENDKEY GENAI CLASSIFICATION QUALITY REPORT",
        "=" * 60,
        f"Total Transactions Processed: {total}",
        f"  - Cached (v2):              {from_cache_count} ({from_cache_count / total * 100:.1f}%)",
        f"  - Newly Classified:         {new_classified} ({new_classified / total * 100:.1f}%)",
        "-" * 60,
        "Confidence Level Distribution:",
        f"  - HIGH Confidence:          {high_conf} ({high_conf / total * 100:.1f}%)",
        f"  - MEDIUM Confidence:        {med_conf} ({med_conf / total * 100:.1f}%)",
        f"  - LOW Confidence:           {low_conf} ({low_conf / total * 100:.1f}%)",
        "-" * 60,
        "Governance & Human-in-the-Loop Status:",
        f"  - ACCEPTED (Auto-pass):     {accepted} ({accepted / total * 100:.1f}%)",
        f"  - REVIEW REQUIRED:          {review_req} ({review_req / total * 100:.1f}%)",
        f"  - INVALID / API Errors:     {invalid} ({invalid / total * 100:.1f}%)",
        f"  - Semantic Mismatches:      {mismatches}",
        "=" * 60,
    ]
    report_text = "\n".join(lines)
    return report_text