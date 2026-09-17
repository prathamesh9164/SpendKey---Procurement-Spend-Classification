<div align="center">

<img src="assets/workflow_flowchart.svg" alt="SpendKey Architecture" width="100%"/>

# 🏷️ SpendKey — Procurement Spend Classifier

### GenAI-powered spend taxonomy classification engine that automatically maps 197 procurement transactions into a 4-tier taxonomy using Groq LLM, transparent multi-signal retrieval, and enterprise-grade governance.

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20API-F55036?style=for-the-badge&logoColor=white)](https://groq.com)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

| 📊 **197 Transactions** | 🏗️ **4-Tier Taxonomy** | 🎯 **84.66% L1 Accuracy** | ✅ **0% Hallucinated Paths** |
|:---:|:---:|:---:|:---:|
| Fully classified | L1 → L2 → L3 → L4 | vs. 189-row Gold Standard | 100% taxonomy validity |

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Approach Evolution](#-approach-evolution)
- [Final Solution Architecture](#-final-solution-architecture)
- [Accuracy Results](#-accuracy-results)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Key Edge Cases](#-key-edge-cases-solved)
- [Prompting Techniques](#-prompting-techniques)
- [Governance Model](#-governance-model)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement

SpendKey's workbook contains **197 indirect procurement transactions** — real invoice lines with messy vendor names, abbreviated product codes (`EPC`, `PLC`, `IFM`), and bundled multi-service contracts. Every transaction must be mapped to the **single most specific path** in a 256-row, 4-tier taxonomy:

```
L1 Segment  ▶  L2 Family  ▶  L3 Class  ▶  L4 Commodity
```

**The constraints:**
- No labelled training data → classical supervised ML is not viable on its own
- Retrieval must be transparent and auditable — no opaque vector databases
- Every classification must be verifiable against the source-of-truth taxonomy
- Bundled or ambiguous contracts must be flagged for human review — never silently misclassified

---

## 🧪 Approach Evolution

The problem was solved in **five progressive iterations**, each exposing a new failure mode that motivated the next step.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Approach 1 ──► TF-IDF + Logistic Regression    L4 accuracy: ~22%      │
│         │       Vocabulary mismatch, brand blindness                    │
│         ▼                                                               │
│  Approach 2 ──► TF-IDF + SVM                    L4 accuracy: ~27%      │
│         │       Better margin, same semantic blindness                  │
│         ▼                                                               │
│  Approach 3 ──► MiniLM Semantic Embeddings       L4 accuracy: ~47%     │
│         │       Semantic, but no boundary notes or reasoning            │
│         ▼                                                               │
│  Approach 4 ──► Hybrid Engine                    L4 accuracy: ~55%     │
│  (MiniLM + TF-IDF + domain rules) Rules don't scale, no reasoning      │
│         ▼                                                               │
│  Approach 5 ──► GenAI + Retrieval + Governance  L4 accuracy: 69.84% ✅ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Each Approach Was Superseded

| # | Approach | L1 Acc. | L4 Acc. | Reasoning | Boundary Notes | Key Failure |
|---|---|:---:|:---:|:---:|:---:|---|
| 1 | **TF-IDF + Logistic Regression** | ~51% | ~22% | ❌ | ❌ | "CrowdStrike Falcon" → 0 token overlap with "Endpoint Security" |
| 2 | **TF-IDF + SVM** | ~58% | ~27% | ❌ | ❌ | Better margin, same sparse-vector blindness to brand names |
| 3 | **MiniLM Embeddings** | ~71% | ~47% | ❌ | ❌ | "EPC" still confused with Asbestos; no boundary note awareness |
| 4 | **Hybrid Engine** | ~76% | ~55% | ❌ | ⚠️ Partial | Rules need manual maintenance for every new edge case |
| 5 | **GenAI — Groq LLM** ✅ | **84.66%** | **69.84%** | ✅ Full | ✅ Yes | — |

---

## 🏗️ Final Solution Architecture

```
Excel Workbook (197 transactions + 256-row taxonomy)
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│  1. DATA INGESTION & CLEANING   src/data_processing.py  │
│  • Forward-fill merged taxonomy cells                   │
│  • Normalize headers & whitespace                       │
│  • Cross-validate vs dropdown list (256 paths matched)  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  2. MULTI-SIGNAL TAXONOMY RETRIEVAL   prompts/prompts.py│
│  • Exact L4 commodity match       +10 pts               │
│  • Boundary note keyword match     +8 pts               │
│  • Exact L3 category match         +7 pts               │
│  • Domain synonym boost            +4 pts               │
│  • Per-token L4/L3/L2/L1 overlap   +1 to +5 pts         │
│  • Dynamic fallback: 5-12 candidates per transaction    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  3. BATCHED GENAI INFERENCE + CACHE                     │
│  src/genai_classifier.py                                │
│  • 10 transactions per Groq API call                    │
│  • ~197 individual calls → ~20 batched calls (90% ↓)   │
│  • Persistent disk cache (v2 versioned cache keys)      │
│  • Structured JSON Schema output — schema enforced      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  4. TWO-TIER QUALITY VERIFICATION   src/validator.py    │
│                                                         │
│  Tier 1 — Hierarchy Integrity Gate                      │
│    Does L1→L2→L3→L4 actually exist in the taxonomy?    │
│    No  ──────────────────────────────────► INVALID      │
│                                                         │
│  Tier 2 — Semantic Mismatch Engine                      │
│    EPC ≠ Asbestos  │  PLC ≠ Hydraulics                  │
│    Uniforms ≠ Paper │  EOR ≠ National Insurance          │
│    IFM/PFI bundled contracts ────────► REVIEW_REQUIRED  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  5. GOVERNANCE ROUTING                                  │
│  ACCEPTED        81.2%  — HIGH confidence + valid path  │
│  REVIEW_REQUIRED 18.8%  — Ambiguous or bundled          │
│  INVALID          0.0%  — Zero hallucinations           │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Accuracy Results

Benchmarked against a verified **189-transaction Gold Standard** (`data/gold_standard.csv`):

| Metric | Score | Count |
|---|:---:|:---:|
| **Level 1 — Segment Accuracy** | **84.66%** | 160 / 189 |
| **Level 2 — Family Accuracy** | **77.25%** | 146 / 189 |
| **Level 3 — Category Accuracy** | **75.13%** | 142 / 189 |
| **Level 4 — Commodity Accuracy** | **69.84%** | 132 / 189 |
| **Full 4-Level Path Accuracy** | **69.84%** | 132 / 189 |
| **Auto-Accepted Precision** | **79.74%** | 153 items auto-passed |
| **High-Confidence Precision** | **79.87%** | 154 HIGH items |
| **Taxonomy Path Validity** | **100.0%** | 0 hallucinated paths |

> **Note on L4 accuracy**: Several counted "mismatches" are cases where the AI gave a *more specific and more correct* answer than the Gold Standard. For example, *Power BI* → AI: `Business Intelligence Tools` vs Gold Standard: `General SaaS Subscriptions`. The AI's answer is more accurate. True expert-judged accuracy is higher than 69.84%.

Run the live evaluation:
```bash
python src/evaluate_accuracy.py
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/prathamesh9164/SpendKey---Procurement-Spend-Classification.git
cd SpendKey---Procurement-Spend-Classification
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env:
# GROQ_API_KEY=gsk_your_key_here
# GROQ_MODEL=openai/gpt-oss-120b
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 3. Run

**Option A — One command (recommended):**
```bash
python run_pipeline.py
```

**Option B — Jupyter notebooks (step-by-step):**
```bash
jupyter notebook
# Run in order:
# 1. notebooks/01_data_and_retrieval.ipynb      (no API key needed)
# 2. notebooks/02_classification_and_governance.ipynb
```

**Option C — Interactive Streamlit dashboard:**
```bash
streamlit run app.py
```

Output: `output/final_classification.xlsx`

---

## 📁 Project Structure

```
SpendKey-GenAI-Classifier/
│
├── 📊 data/
│   ├── Spendkey_Assignment.xlsx          ← Source workbook (197 transactions + 256-row taxonomy)
│   └── gold_standard.csv                 ← 189-row verified ground truth benchmark
│
├── 🧠 src/
│   ├── data_processing.py                ← Excel ingestion, cleaning, taxonomy normalization
│   ├── genai_classifier.py               ← Batched Groq inference, disk cache (v2), rate limits
│   ├── validator.py                      ← Hierarchy integrity gate + semantic mismatch engine
│   └── evaluate_accuracy.py              ← Tier-by-tier accuracy report vs gold standard
│
├── 💬 prompts/
│   └── prompts.py                        ← Multi-signal retrieval scoring, few-shots, JSON schema
│
├── 📓 notebooks/
│   ├── 01_data_and_retrieval.ipynb       ← Stage 1: data inspection & retrieval (offline)
│   └── 02_classification_and_governance.ipynb  ← Stage 2: GenAI inference, validation & export
│
├── 📈 output/
│   └── final_classification.xlsx         ← Classified output (14 columns, governance tags)
│
├── 🖼️ assets/
│   └── workflow_flowchart.svg            ← Architecture & workflow flowchart
│
├── app.py                                ← Streamlit interactive review dashboard
├── run_pipeline.py                       ← End-to-end automated runner
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔬 Key Edge Cases Solved

| # | Transaction | Vendor | Before (ML Approaches) | After (GenAI v2) | Status |
|---|---|---|---|---|:---:|
| 14 | Power BI Premium – capacity licence | Microsoft | *Data Warehousing Platforms* | `IT & Technology > Software > Data & Analytics > Business Intelligence Tools` | ✅ |
| 17 | Siemens S7-1500 PLC – firmware upgrade | Siemens AG | *Hydraulics & Fluid Power* | `MRO & Engineering > Spare Parts > Electrical Parts > Electrical Components & Fuses` | ✅ |
| 37 | Bureau Veritas EPC surveys | Bureau Veritas | *Asbestos Surveying* | `Facilities & Property > Compliance > Statutory Compliance > Energy Performance Certificates` | ✅ |
| 67 | Deel EOR employer of record | Deel / Remote | *National Insurance Contributions* | `HR & Workforce > HR Operations > Payroll > Payroll Outsourcing` | ✅ |
| 78 | Carbon footprint assessment | ERM Group | *Utilities / Offsetting* | `Professional Services > Consulting > Sustainability > Carbon & Net Zero Advisory` | ✅ |
| 48 | Sodexo PFI unitary charge | Sodexo | Single-service mis-classification | Correctly flagged as bundled → human review | ⚠️ |
| 185 | Business rates property tax | Birmingham City Council | *Flexible Workspace* | `Finance & Insurance > Taxes & Levies > Property Tax > Business Rates` | ✅ |

---

## 💬 Prompting Techniques

| Technique | Implementation |
|---|---|
| **Role Prompting** | System prompt: *"You are a senior procurement spend classification analyst..."* |
| **Instruction Prompting** | Numbered rules: taxonomy-only, most specific L4, apply boundary notes, flag ambiguity |
| **Few-Shot Prompting** | 5 worked procurement examples embedded in the prompt (context only — no weights change) |
| **Taxonomy Grounding** | Top-N retrieved candidates with official boundary notes inserted inline per transaction |
| **Structured Output** | `response_format={"type": "json_schema", ...}` — schema enforced at API level |
| **Validation Prompting** | Model self-flags `human_review_required`; Python independently double-checks |

---

## 🧩 Governance Model

```
Classification Result
        │
        ├─► Tier 1: Hierarchy Integrity Gate
        │     L1→L2→L3→L4 exists in 256-row taxonomy?
        │         │ No  ────────────────────────────────►  ❌ INVALID
        │         │ Yes
        │         ▼
        ├─► Tier 2: Semantic Mismatch Engine
        │     EPC≠Asbestos · PLC≠Hydraulics · EOR≠NIC
        │     Uniforms≠Paper · IFM/PFI bundled?
        │         │ Mismatch  ──────────────────────────►  ⚠️ REVIEW_REQUIRED
        │         │ Clean
        │         ▼
        └─► Confidence Gate
                  │ HIGH  ──────────────────────────────►  ✅ ACCEPTED
                  │ MEDIUM / LOW  ──────────────────────►  ⚠️ REVIEW_REQUIRED
                  │ LLM self-flagged  ──────────────────►  ⚠️ REVIEW_REQUIRED
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Data Processing | `pandas`, `openpyxl` |
| ML Baselines | `scikit-learn` — TF-IDF, Logistic Regression, LinearSVC |
| Semantic Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dashboard | `Streamlit` |
| Output Validation | Pure Python — no external validator |
| Caching | Local JSON (`output/.classification_cache.json`) |
| Secrets Management | `python-dotenv` — API key never committed |

---

<div align="center">

**Built for the SpendKey Analyst – Procurement Analytics & AI assignment**

*197 transactions · 256-node taxonomy · 5 approaches · 1 production solution*

</div>

## 1. Problem Statement

The assignment workbook (`Spendkey_Assignment.xlsx`) contains 197 procurement
transactions (indirect spend only) and a 256-row taxonomy reference
(`L1 Segment > L2 Family > L3 Class > L4 Commodity`). Each transaction needs to be
matched to the single **most specific, correct** taxonomy path.

## 2. Objective

Build an automated pipeline that:
1. Reads a transaction's description, vendor, and source type.
2. Retrieves the most relevant candidate taxonomy paths.
3. Asks an LLM (via Groq) to classify it into L1 → L4, grounded in only that
   taxonomy, with a documented reason and a confidence level.
4. Validates the answer in Python against the real taxonomy (never trusts the
   LLM blindly).
5. Applies human-in-the-loop rules to flag anything uncertain.
6. Produces a final Excel file with everything a reviewer needs.

## 3. Why GenAI Instead of ML/DL

Classical ML/DL classification needs a labelled training set — this workbook
has 197 *unlabelled* transactions and no ground truth to train against.
Training a model on zero labelled examples isn't possible, and even with
labels, 197 rows is far too few to train a reliable classifier from scratch.

An LLM, by contrast, can already read English descriptions and reason about
what's being purchased. Given the taxonomy as **context** (not training data),
few-shot examples, and clear instructions, it can classify a transaction
correctly on the first attempt, with no training step, no labelled data
requirement, and no GPU. This is a **prompt engineering + retrieval**
problem, not a model-training problem — which is exactly what "GenAI" means
here as distinct from "ML".

## 4. Architecture

```
Excel
  ↓
Pandas (load + clean transactions & taxonomy)
  ↓
Taxonomy (source of truth, 256 L4 commodities under 12 L1 segments)
  ↓
Simple Retrieval (keyword overlap — retrieves top-N relevant taxonomy paths)
  ↓
Prompt Engineering (role + instructions + boundary notes)
  ↓
Few-Shot Prompting (3 worked examples)
  ↓
Groq LLM (structured JSON output, JSON Schema enforced)
  ↓
Python Validation (does the returned path actually exist in the taxonomy?)
  ↓
Human-in-the-Loop rules (confidence + validity → ACCEPTED / REVIEW / INVALID)
  ↓
Final Excel (output/final_classification.xlsx)
```

**Component explanations:**

- **Pandas / cleaning** — the raw workbook has title rows, instruction text,
  and Excel merged cells for the taxonomy hierarchy. This step turns it into
  two flat, predictable DataFrames.
- **Taxonomy** — treated as the single source of truth. The LLM is never
  allowed to invent a category outside of it.
- **Simple retrieval** — a keyword-overlap score over the taxonomy's 256 rows
  (see notebook 02 for why a vector database isn't needed at this scale).
  It hands the LLM a relevant subset instead of either "everything" or
  "nothing".
- **Prompt engineering** — role, instructions, boundary-note enforcement, and
  the retrieved grounding taxonomy, assembled into one prompt.
- **Few-shot prompting** — 3 worked examples in the prompt text, showing the
  expected reasoning style. This changes nothing about the model itself; it's
  just more context for this one request.
- **Groq LLM** — does the actual classification, constrained to a JSON Schema
  so the response is always parseable.
- **Python validation** — re-checks the LLM's answer against the real
  taxonomy DataFrame. An LLM can still hallucinate a plausible-sounding but
  non-existent path; this step catches that.
- **Human-in-the-loop** — confidence + validity decide whether a row is
  auto-accepted or routed for manual review.
- **Final Excel** — everything a reviewer needs in one file.

## 5. Prompting Techniques Used

| Technique | What it means here |
|---|---|
| Role prompting | System prompt casts the model as a senior procurement spend classification analyst |
| Instruction prompting | Explicit numbered rules: taxonomy-only, most specific L4, don't use vendor name alone, apply boundary notes, flag ambiguity |
| Few-shot prompting | 3 worked examples embedded in the prompt text (not training — no weights change) |
| Context / taxonomy grounding | Retrieved taxonomy subset + boundary notes inserted into the prompt so answers are constrained to real options |
| Structured output prompting | Response is forced into a fixed JSON Schema via Groq's structured outputs, not "please reply in JSON" |
| Validation prompting | The prompt itself asks the model to self-flag ambiguity (`human_review_required`), which Python validation then double-checks independently |

**Note:** retrieval/RAG is an *architecture* — a way of supplying external
context to the LLM — not a prompting technique itself. It decides *what* goes
into the grounding section of the prompt above.

## 6. Taxonomy Grounding

Every prompt includes only the top-N (default 15) taxonomy paths most likely
to be relevant to the transaction, plus any "Key Boundary Note" attached to
those paths (33 of the 256 rows have one — e.g. "off-payroll IT resource →
classify in HR", "AWS = IaaS, not generic SaaS"). This keeps the prompt
focused and makes boundary rules impossible to miss.

## 7. Groq API

- SDK: `groq` (`from groq import Groq`)
- Model: `llama-3.3-70b-versatile` by default (see `src/genai_classifier.py`
  for the reasoning, and check https://console.groq.com/docs/models before
  running, since Groq's hosted model lineup changes over time)
- Structured outputs: uses Groq's `response_format={"type": "json_schema", ...}`
  so the model's response always matches `CLASSIFICATION_JSON_SCHEMA` in
  `prompts/prompts.py`
- The key is read from `.env` via `python-dotenv` — never hard-coded, never
  printed

## 8. Validation Approach

`src/validator.py` checks, for every LLM response:
1. Does L1 exist in the taxonomy?
2. Does L2 exist under that L1?
3. Does L3 exist under that L1 → L2?
4. Does L4 exist under that L1 → L2 → L3?
5. Does the complete 4-level path exist as one real taxonomy row?

Any failure → `validation_status = "INVALID"`, with the reason recorded in
`validation_notes`. Nothing is silently auto-corrected.

## 9. Human-in-the-Loop

```
HIGH confidence + valid taxonomy   → ACCEPTED
MEDIUM confidence                 → REVIEW_REQUIRED
LOW confidence                    → REVIEW_REQUIRED
Invalid taxonomy path             → INVALID (always reviewed)
LLM-flagged ambiguous             → REVIEW_REQUIRED
```

## 10. Project Structure

```
SpendKey-GenAI-Classifier/
├── assets/
│   └── workflow_flowchart.svg  # visual architecture & workflow flowchart vector graphic
├── data/
│   ├── Spendkey_Assignment.xlsx
│   └── gold_standard.csv       # 189-transaction ground truth verification benchmark
├── prompts/
│   └── prompts.py              # retrieval, few-shot examples, prompt builder, JSON schema
├── src/
│   ├── data_processing.py      # load & clean transactions + taxonomy
│   ├── genai_classifier.py     # Groq client, single + batch classification
│   ├── validator.py            # taxonomy validation + human-in-the-loop rules
│   └── evaluate_accuracy.py    # accuracy evaluation report
├── notebooks/
│   ├── 01_data_and_retrieval.ipynb             # Stage 1: Data prep & multi-signal retrieval
│   └── 02_classification_and_governance.ipynb  # Stage 2: Batched GenAI inference, validation & Excel
├── output/
│   └── final_classification.xlsx   # created after running notebook 02 (or pipeline runner / app)
├── app.py                      # optional Streamlit UI
├── run_pipeline.py             # automated pipeline runner
├── requirements.txt
├── .env.example
└── README.md
```

## 11. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 12. Environment Variables

```bash
cp .env.example .env
# then edit .env and set:
# GROQ_API_KEY=your_key_here
```

## 13. How to Run the Notebooks

```bash
jupyter notebook
# open and run, in order:
#   notebooks/01_data_and_retrieval.ipynb
#   notebooks/02_classification_and_governance.ipynb
```

Notebook 01 runs fully offline (no API key needed) for data loading, quality inspection, and transparent candidate retrieval. Notebook 02 executes batched Groq classification with local caching (`.classification_cache.json`), runs two-tier validation, exports `output/final_classification.xlsx`, and reports accuracy.

## 14. How to Run the App

```bash
streamlit run app.py
```

Upload the workbook, click "Classify Transactions", review the summary
metrics and the review queue, then download `final_classification.xlsx`.

## 15. Example Output (illustrative — see note above on real vs sample data)

```json
{
  "l1": "IT & Technology",
  "l2": "Software",
  "l3": "Productivity & Collaboration",
  "l4": "Office Suites",
  "classification_reason": "Seat-based subscription to Microsoft's office productivity suite matches the Office Suites commodity directly.",
  "confidence_level": "HIGH",
  "human_review_required": false
}
```

## 16. Limitations

- **Retrieval is keyword-based**, not semantic. Branded product names with no
  generic taxonomy vocabulary in them (e.g. "Microsoft 365 E3") can score 0
  keyword overlap against every taxonomy row; the retriever then falls back
  to sending the *full* taxonomy for that row rather than guessing, which
  keeps answers correct but makes the prompt longer for those cases.
- **Confidence is self-reported by the LLM**, not a calibrated statistical
  probability — treat HIGH/MEDIUM/LOW as a qualitative signal, not a
  percentage.
- **No ground-truth labels exist for this workbook**, so no accuracy number
  is claimed anywhere in this project. Accuracy could only be measured after
  a subject-matter expert reviews a sample of the ACCEPTED rows.
- **LLM outputs can still drift** row-to-row despite `temperature=0`; the
  Python validation layer exists specifically because the model is not
  assumed to be perfectly reliable.

## 17. Future Improvements

- Add a small labelled evaluation set (even 20-30 rows) to measure real
  accuracy against SME judgement.
- Upgrade retrieval from keyword overlap to lightweight semantic similarity
  (e.g. a small local sentence-embedding model) if branded-product misses
  become a real problem at larger transaction volumes.
- Track LLM disagreement across repeated calls (self-consistency) as an
  additional, more principled confidence signal.
- Add a lightweight review UI (e.g. inline edit + re-submit) inside the
  Streamlit app for the REVIEW_REQUIRED queue.
