"""
prompts.py

Everything related to:
  1. Simple, transparent taxonomy retrieval (no vector DB needed for 256 rows).
  2. Few-shot examples.
  3. Building the final prompt sent to the Groq LLM.

WHY SIMPLE RETRIEVAL INSTEAD OF A VECTOR DATABASE?
---------------------------------------------------
The taxonomy has only 256 leaf-level rows. A full vector database (FAISS,
Chroma, etc.) is built for retrieval over thousands-to-millions of documents,
where you cannot afford to scan everything. At 256 short text rows, a plain
keyword-overlap score over every row runs in milliseconds and is fully
transparent/debuggable -- you can print exactly why a taxonomy path was
retrieved. Introducing embeddings + a vector index here would add complexity
without solving a real performance problem, so we use simple keyword
retrieval, which is the RAG-style "architecture" the assignment asks for:
it supplies the LLM with a *relevant, taxonomy-grounded subset* of context
instead of either (a) the whole taxonomy every time or (b) no taxonomy at all.
"""

import re
from collections import Counter

from src.data_processing import TAX_L1, TAX_L2, TAX_L3, TAX_L4, TAX_FULL_PATH, TAX_BOUNDARY_NOTE


# ---------------------------------------------------------------------------
# 1. SIMPLE RETRIEVAL (keyword overlap, no vector DB)
# ---------------------------------------------------------------------------

# Common words that carry no classification signal; ignoring them keeps the
# keyword overlap score meaningful.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "to", "in", "on", "with",
    "annual", "monthly", "quarterly", "renewal", "licence", "license",
    "units", "unit", "seats", "seat", "billing", "programme", "program",
    "agreement", "contract", "service", "services",
}


_DOMAIN_SYNONYMS = {
    # IT, Software & Cloud
    "microsoft": {"software", "productivity", "office", "collaboration", "cloud"},
    "m365": {"software", "productivity", "office", "collaboration"},
    "365": {"software", "productivity", "office", "collaboration"},
    "aws": {"cloud", "iaas", "compute", "storage", "infrastructure"},
    "azure": {"cloud", "iaas", "compute", "storage", "infrastructure"},
    "google": {"cloud", "software", "analytics", "advertising"},
    "salesforce": {"software", "crm", "applications"},
    "workday": {"software", "hr", "payroll", "applications"},
    "servicenow": {"software", "itsm", "applications"},
    "slack": {"software", "collaboration", "messaging"},
    "zoom": {"software", "collaboration", "conferencing"},
    "sap": {"software", "erp", "applications"},
    "oracle": {"software", "database", "erp"},
    "iaas": {"cloud", "compute", "infrastructure"},
    "saas": {"software", "cloud", "applications"},
    "bi": {"business", "intelligence", "tools", "analytics", "data", "software"},
    "power bi": {"business", "intelligence", "tools", "analytics", "software"},
    "tableau": {"business", "intelligence", "tools", "analytics", "software"},
    "digital twin": {"software", "engineering", "applications", "consulting"},

    # Energy, Environmental & Sustainability
    "epc": {"energy", "performance", "certificates", "statutory", "compliance"},
    "energy performance": {"energy", "performance", "certificates", "statutory", "compliance"},
    "carbon": {"carbon", "net", "zero", "advisory", "offsetting", "sustainability"},
    "scope 1": {"carbon", "net", "zero", "advisory", "sustainability"},
    "scope 3": {"carbon", "net", "zero", "advisory", "sustainability"},
    "footprint": {"carbon", "net", "zero", "advisory", "sustainability"},
    "erm": {"carbon", "net", "zero", "advisory", "sustainability"},
    "bureau veritas": {"energy", "performance", "certificates", "statutory", "compliance"},

    # Engineering, MRO & Electrical
    "plc": {"electrical", "components", "fuses", "automation", "parts"},
    "scada": {"electrical", "components", "automation", "control", "parts"},
    "siemens": {"electrical", "components", "fuses", "parts", "automation"},
    "lubricant": {"industrial", "lubricants", "oils", "lubrication", "chemicals"},
    "lubricants": {"industrial", "lubricants", "oils", "lubrication", "chemicals"},
    "hydraulic": {"hydraulics", "fluid", "power", "industrial", "lubricants", "oils"},
    "hydraulic fluid": {"industrial", "lubricants", "oils", "lubrication", "chemicals"},
    "fuchs": {"industrial", "lubricants", "oils", "lubrication"},

    # Workwear, PPE & HR
    "uniform": {"uniforms", "branded", "clothing", "workwear", "ppe"},
    "uniforms": {"uniforms", "branded", "clothing", "workwear", "ppe"},
    "workwear": {"uniforms", "branded", "clothing", "workwear", "ppe"},
    "scrubs": {"uniforms", "branded", "clothing", "workwear", "medical"},
    "simon jersey": {"uniforms", "branded", "clothing", "workwear"},
    "ppe": {"safety", "ppe", "helmets", "gloves", "boots", "workwear"},
    "helmets": {"safety", "ppe", "helmets", "gloves", "boots"},
    "boots": {"safety", "ppe", "helmets", "gloves", "boots"},
    "gloves": {"safety", "ppe", "helmets", "gloves", "boots"},
    "rs components": {"safety", "ppe", "helmets", "gloves", "boots", "electrical"},
    "locum": {"temporary", "contract", "labour", "clinical", "medical", "staffing", "contractors"},
    "doctors direct": {"temporary", "contract", "labour", "clinical", "medical"},
    "ir35": {"contractors", "temporary", "contract", "labour", "professional"},
    "recruiter": {"recruitment", "contingency", "permanent", "executive", "search"},
    "recruitment": {"recruitment", "contingency", "permanent", "executive", "search"},
    "headhunter": {"recruitment", "executive", "search", "permanent"},
    "rpo": {"recruitment", "process", "outsourcing", "contingency", "permanent"},
    "eor": {"payroll", "outsourcing", "hr", "operations"},
    "employer of record": {"payroll", "outsourcing", "hr", "operations"},
    "deel": {"payroll", "outsourcing", "hr"},
    "cipd": {"professional", "body", "memberships", "corporate", "services"},
    "membership": {"professional", "body", "memberships", "corporate", "services", "subscriptions"},
    "memberships": {"professional", "body", "memberships", "corporate", "services", "subscriptions"},

    # Professional Services, Legal & Finance
    "actuarial": {"actuarial", "valuations", "pension", "finance", "accounting"},
    "pension": {"actuarial", "valuations", "pension", "payroll"},
    "mercer": {"actuarial", "valuations", "pension"},
    "rates": {"business", "rates", "property", "tax", "taxes", "levies"},
    "business rates": {"business", "rates", "property", "tax", "taxes", "levies"},
    "deloitte": {"consulting", "advisory", "audit"},
    "pwc": {"consulting", "advisory", "audit", "tax"},
    "kpmg": {"consulting", "advisory", "audit", "tax"},
    "ey": {"consulting", "advisory", "audit", "tax"},
    "mckinsey": {"consulting", "strategy", "advisory"},

    # Marketing, Brand & Events
    "ooh": {"brand", "strategy", "design", "advertising", "creative"},
    "employer brand": {"brand", "strategy", "design", "recruitment", "advertising"},
    "wtw": {"brand", "strategy", "design", "creative", "advertising"},
    "conference": {"conference", "event", "management", "events", "sponsorship"},
    "event": {"conference", "event", "management", "events", "sponsorship"},

    # Logistics & Facilities
    "returns": {"returns", "processing", "reverse", "logistics", "courier", "parcel"},
    "reverse logistics": {"returns", "processing", "reverse", "logistics"},
    "clipper": {"returns", "processing", "reverse", "logistics", "warehousing"},
    "ifm": {"facilities", "management", "hard", "fm", "cleaning", "catering"},
    "pfi": {"facilities", "property", "hard", "fm", "unitary"},
    "asbestos": {"asbestos", "surveying", "management", "statutory", "compliance"},
    "legionella": {"water", "treatment", "statutory", "compliance"},
}


def _tokenize(text: str) -> list:
    """Lowercase, strip punctuation, split into words, drop stopwords."""
    text = text.lower()
    words = re.findall(r"[a-z0-9&]+", text)
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def retrieve_relevant_taxonomy(transaction_text: str, taxonomy_df, top_n: int = 7):
    """
    Multi-signal, explainable taxonomy retrieval:
    1. Exact L4 commodity name match in transaction text (+10)
    2. Exact L3 category name match in transaction text (+7)
    3. Boundary note keyword match (+8 per matched token)
    4. L4 token overlap (+5 per token)
    5. L3 token overlap (+3 per token)
    6. L2 token overlap (+2 per token)
    7. L1 token overlap (+1 per token)
    8. Domain synonyms & vendor hints expansion (+4 per match)

    Dynamic two-stage fallback:
    - If strong match (score >= 8): top 5-7 focused candidates
    - If moderate match (score 3-7): top 10-12 broader candidates
    - If weak/zero match (< 3): cross-section of 2 per L1
    """
    text_lower = transaction_text.lower()
    raw_tokens = set(_tokenize(transaction_text))
    query_tokens = set(raw_tokens)

    # Single-word domain expansions
    for token in raw_tokens:
        if token in _DOMAIN_SYNONYMS:
            query_tokens.update(_DOMAIN_SYNONYMS[token])

    # Multi-word phrase expansions
    for phrase, syns in _DOMAIN_SYNONYMS.items():
        if " " in phrase and phrase in text_lower:
            query_tokens.update(syns)

    scores = []
    for _, row in taxonomy_df.iterrows():
        l4_str = str(row[TAX_L4]).lower()
        l3_str = str(row[TAX_L3]).lower()
        l2_str = str(row[TAX_L2]).lower()
        l1_str = str(row[TAX_L1]).lower()
        note_str = str(row[TAX_BOUNDARY_NOTE]).lower() if row[TAX_BOUNDARY_NOTE] else ""

        l4_tokens = set(_tokenize(l4_str))
        l3_tokens = set(_tokenize(l3_str))
        l2_tokens = set(_tokenize(l2_str))
        l1_tokens = set(_tokenize(l1_str))
        note_tokens = set(_tokenize(note_str))

        score = 0
        # Exact L4 phrase match
        if l4_str and l4_str in text_lower:
            score += 10
        # Exact L3 phrase match
        if l3_str and l3_str in text_lower:
            score += 7

        # Boundary note keyword matches
        matched_notes = query_tokens & note_tokens
        score += 8 * len(matched_notes)

        # Token overlap scores
        score += 5 * len(query_tokens & l4_tokens)
        score += 3 * len(query_tokens & l3_tokens)
        score += 2 * len(query_tokens & l2_tokens)
        score += 1 * len(query_tokens & l1_tokens)

        scores.append(score)

    tax_copy = taxonomy_df.copy()
    tax_copy["_retrieval_score"] = scores

    max_score = tax_copy["_retrieval_score"].max()
    if max_score >= 8:
        effective_n = min(top_n, 7)
    elif max_score >= 3:
        effective_n = min(max(top_n, 10), 12)
    else:
        # Fallback: cross-section across L1
        return tax_copy.groupby(TAX_L1).head(2).drop(columns="_retrieval_score")

    top = tax_copy.sort_values("_retrieval_score", ascending=False).head(effective_n)
    return top.drop(columns="_retrieval_score")


def format_taxonomy_for_prompt(taxonomy_subset_df) -> str:
    """Render a taxonomy subset as compact, LLM-readable lines with attached boundary notes."""
    lines = []
    for _, row in taxonomy_subset_df.iterrows():
        line = f"- {row[TAX_FULL_PATH]}"
        if row[TAX_BOUNDARY_NOTE] and str(row[TAX_BOUNDARY_NOTE]).strip():
            line += f"  [Boundary note: {row[TAX_BOUNDARY_NOTE]}]"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. FEW-SHOT EXAMPLES
# ---------------------------------------------------------------------------
# Curated directly from the SpendKey taxonomy and real indirect procurement rules:
FEW_SHOT_EXAMPLES = [
    {
        "spend_description": "Contract .NET developer, 6-month engagement, off-payroll",
        "vendor": "Harvey Nash",
        "classification": {
            "l1": "HR & Workforce",
            "l2": "Temporary & Contract Labour",
            "l3": "Professional Contractors",
            "l4": "IT Contractors",
            "classification_reason": "Off-payroll IT contractor labour classifies under HR & Workforce per boundary note, not IT Software.",
            "confidence_level": "HIGH",
            "human_review_required": False,
        },
    },
    {
        "spend_description": "Siemens S7-1500 PLC – firmware upgrade + engineering support days",
        "vendor": "Siemens AG",
        "classification": {
            "l1": "MRO & Engineering",
            "l2": "Spare Parts & Components",
            "l3": "Electrical Parts",
            "l4": "Electrical Components & Fuses",
            "classification_reason": "Programmable logic controllers (PLCs) are electronic automation controls, classifying under Electrical Parts, not mechanical hydraulics.",
            "confidence_level": "HIGH",
            "human_review_required": False,
        },
    },
    {
        "spend_description": "Bureau Veritas – EPC surveys, full estate, 40 sites",
        "vendor": "Bureau Veritas",
        "classification": {
            "l1": "Facilities & Property",
            "l2": "Compliance & Environment",
            "l3": "Statutory Compliance",
            "l4": "Energy Performance Certificates",
            "classification_reason": "Statutory energy efficiency compliance rating for building estates, distinct from environmental hazardous waste/asbestos.",
            "confidence_level": "HIGH",
            "human_review_required": False,
        },
    },
    {
        "spend_description": "Simon Jersey – pharmacy staff uniform refresh, 2,000 sets",
        "vendor": "Simon Jersey",
        "classification": {
            "l1": "HR & Workforce",
            "l2": "Workwear & PPE",
            "l3": "Workwear",
            "l4": "Uniforms & Branded Clothing",
            "classification_reason": "Staff uniforms and corporate branded apparel classify under Workwear & PPE, never office stationery.",
            "confidence_level": "HIGH",
            "human_review_required": False,
        },
    },
    {
        "spend_description": "Deel EOR – employer of record, 25 employees, 12 countries",
        "vendor": "Deel / Remote",
        "classification": {
            "l1": "HR & Workforce",
            "l2": "HR Operations",
            "l3": "Payroll",
            "l4": "Payroll Outsourcing",
            "classification_reason": "Employer of Record is a managed global employment and payroll compliance service, not internal tax remittance.",
            "confidence_level": "HIGH",
            "human_review_required": False,
        },
    },
]


def format_few_shot_examples() -> str:
    """Render the few-shot examples as text blocks for the prompt."""
    blocks = []
    for ex in FEW_SHOT_EXAMPLES:
        c = ex["classification"]
        blocks.append(
            f"Transaction:\n"
            f"  Description: {ex['spend_description']}\n"
            f"  Vendor: {ex['vendor']}\n"
            f"Classification:\n"
            f"  Path: {c['l1']} > {c['l2']} > {c['l3']} > {c['l4']}\n"
            f"  Reason: {c['classification_reason']}\n"
            f"  Confidence: {c['confidence_level']}\n"
            f"  Human Review Required: {str(c['human_review_required']).lower()}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 3. SYSTEM / ROLE PROMPT (role prompting + instruction prompting)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior procurement spend classification analyst specializing in indirect spend taxonomy mapping.

Your job is to classify procurement transactions into the exact SpendKey taxonomy path (L1 -> L2 -> L3 -> L4) using ONLY the taxonomy options provided for each transaction.

Binding Rules:
1. Grounding: Select ONLY from the candidate taxonomy paths listed for that transaction. Never invent a category or modify taxonomy wording.
2. Specificity: Select the most specific valid L4 commodity that accurately describes the product or service being acquired.
3. Primacy of Purchase Description: Base classification on what is actually being purchased, not supplier branding alone (e.g. Siemens sells engineering software, medical devices, and PLCs; Bureau Veritas provides EPCs, asbestos tests, and inspections).
4. Boundary Notes are Authoritative: When a candidate path has a [Boundary note], that note is a binding rule that overrides generic assumptions (e.g., IT Contractor labour must be classified in HR & Workforce, not IT & Technology).
5. Confidence Guidelines:
   - HIGH: The purchase description directly corresponds to the L4 commodity definition.
   - MEDIUM: Plausible match with minor ambiguity between closely related categories.
   - LOW: Weak fit, or missing exact leaf category.
6. Human Review Required:
   - Set human_review_required = false for clear, unambiguous single-commodity matches.
   - Set human_review_required = true for:
     a) Bundled multi-service contracts (e.g. Integrated Facilities Management - IFM, PFI Unitary Charges) where multiple distinct services are grouped in one invoice.
     b) Specialized staffing where the taxonomy lacks a specific category (e.g., Doctors Direct clinical locum temps).
     c) Any transaction where no candidate path accurately represents the purchase.
7. Output Format: Return valid JSON strictly adhering to the specified schema.
"""


def build_classification_prompt(transaction_row, taxonomy_subset_df) -> str:
    """
    Assemble the single-turn prompt: transaction details + retrieved
    taxonomy (grounding) + few-shot examples + structured output instructions.
    """
    taxonomy_block = format_taxonomy_for_prompt(taxonomy_subset_df)
    examples_block = format_few_shot_examples()

    prompt = f"""Below are EXAMPLES of correctly classified transactions illustrating expected reasoning:

{examples_block}

--------------------------------------------------------------------------
CANDIDATE TAXONOMY PATHS FOR THIS TRANSACTION
--------------------------------------------------------------------------
{taxonomy_block}

--------------------------------------------------------------------------
TRANSACTION TO CLASSIFY
--------------------------------------------------------------------------
Spend Description: {transaction_row['spend_description']}
Vendor / Supplier: {transaction_row['vendor']}
Source Type: {transaction_row['source_type']}

--------------------------------------------------------------------------
INSTRUCTIONS
--------------------------------------------------------------------------
Select the most accurate L1 > L2 > L3 > L4 path from the candidate list above.
If the spend is a bundled contract or lacks an exact category, select the closest category and set human_review_required to true.
"""
    return prompt


# JSON Schema for Groq structured outputs (json_schema response format).
CLASSIFICATION_JSON_SCHEMA = {
    "name": "spend_classification",
    "schema": {
        "type": "object",
        "properties": {
            "l1": {"type": "string"},
            "l2": {"type": "string"},
            "l3": {"type": "string"},
            "l4": {"type": "string"},
            "classification_reason": {"type": "string"},
            "confidence_level": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            "human_review_required": {"type": "boolean"},
        },
        "required": [
            "l1", "l2", "l3", "l4",
            "classification_reason", "confidence_level", "human_review_required",
        ],
        "additionalProperties": False,
    },
}

def build_batch_classification_prompt(rows, contexts):
    """
    Build a batched classification prompt containing multiple transactions.
    Each transaction includes its own tailored candidate taxonomy paths with attached boundary notes.
    """
    sections = []

    for row, context in zip(rows, contexts):
        paths_list = context.get("paths", [])
        taxonomy = "\n".join(f"- {path}" for path in paths_list)

        sections.append(
            f"""TRANSACTION ID: {int(row['transaction_id'])}
Spend Description: {row['spend_description']}
Vendor: {row['vendor']}
Source Type: {row['source_type']}

CANDIDATE TAXONOMY PATHS:
{taxonomy}
"""
        )

    examples_block = format_few_shot_examples()

    return f"""You are a senior procurement spend classification analyst classifying transactions into the SpendKey taxonomy.

KEY PROCUREMENT CLASSIFICATION RULES:
1. Use ONLY the candidate taxonomy paths supplied under each transaction.
2. Select the most specific valid L4 commodity.
3. Classify based primarily on what is being purchased, using vendor name as context.
4. Apply [Boundary notes] as binding authoritative rules (e.g. IT contractors go to HR & Workforce).
5. Bundled contracts (e.g. IFM bundled cleaning/catering, PFI Unitary Charge) or unmapped medical locum staffing must be assigned the closest category with human_review_required=true.
6. Set confidence_level to HIGH, MEDIUM, or LOW based on evidence.
7. Return exactly one classification for every transaction in the batch, preserving transaction_id.

FEW-SHOT EXAMPLES:
{examples_block}

TRANSACTIONS TO CLASSIFY:
{"--------------------------------------------------------------------------\n".join(sections)}
"""

BATCH_CLASSIFICATION_JSON_SCHEMA = {
    "name": "spend_classification_batch",
    "schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "integer"
                        },
                        "l1": {
                            "type": "string"
                        },
                        "l2": {
                            "type": "string"
                        },
                        "l3": {
                            "type": "string"
                        },
                        "l4": {
                            "type": "string"
                        },
                        "classification_reason": {
                            "type": "string"
                        },
                        "confidence_level": {
                            "type": "string",
                            "enum": [
                                "HIGH",
                                "MEDIUM",
                                "LOW"
                            ]
                        },
                        "human_review_required": {
                            "type": "boolean"
                        }
                    },
                    "required": [
                        "transaction_id",
                        "l1",
                        "l2",
                        "l3",
                        "l4",
                        "classification_reason",
                        "confidence_level",
                        "human_review_required"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": ["classifications"],
        "additionalProperties": False
    }
}