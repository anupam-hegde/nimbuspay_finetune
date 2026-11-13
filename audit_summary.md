# NimbusPay Synthetic Dataset Audit & Cleaning Summary

This report documents the systematic audit, diagnosis, and remediation of `data/train.jsonl` (1,445 synthetic support tickets) for NimbusPay ticket triage, executed in strict accordance with `SCHEMA.md` and `score.py`.

## 1. Executive Summary & Pipeline Metrics

| Metric | Count | Percentage |
|---|---|---|
| **Total Raw Training Records** | **1,445** | 100.0% |
| **Dropped Records (`drop`)** | **96** | 6.6% |
| **Remediated Records (`fix`)** | **175** | 12.1% |
| **Untouched Baseline Clean Records** | **1174** | 81.2% |
| **Final Clean Training Dataset (`data/train_clean.jsonl`)** | **1349** | 93.4% |

### Validation Scorecard
- **Cleaned Dataset Schema Compliance:** **100.0%** across all 8 canonical keys (`category`, `priority`, `amount_paise`, `txn_date`, `txn_id`, `channel`, `language`, `needs_human`).
- **Missing or Extra Keys:** **0.0%**.
- **Type Compliance:** **100.0%** (exact Python type matching for booleans, integers, strings, and nulls under `score.py` evaluation mechanics).

---

## 2. Defect Categories & Distributions

The audit detected multiple orthogonal failure modes introduced during synthetic ticket generation and formatting:

| Rank | Defect Category | Affected Rows | Share of Data | Action | Remediation Strategy |
|---|---|---|---|---|---|
| 1 | **Key Typo (`prio`) & Invalid Enum Values (`low`/`medium`/`high`)** | 70 | 4.8% | `fix` | Renamed `prio` to `priority`; mapped legacy values (`low` → `P3`, `medium` → `P2`, `high` → `P1`). |
| 2 | **Exact Duplicate Ticket Text** | 61 | 4.2% | `drop` | Dropped identical duplicate user tickets to prevent distribution skew and overfitting; preserved first occurrence. |
| 3 | **Priority Tier Logic Inconsistency (`P1`/`P2`/`P3` Conflict)** | 60 | 4.2% | `fix` | Recomputed deterministic tier (`P1`, `P2`, `P3`) using legal threat keywords, fraud classification, and repaired transaction amount thresholds. |
| 4 | **Non-standard Transaction Date Format (`DD/MM/YYYY`)** | 30 | 2.1% | `fix` | Parsed date strings and converted to standard ISO 8601 `YYYY-MM-DD`. |
| 5 | **Truncated Assistant Response / Malformed JSON** | 25 | 1.7% | `drop` | Dropped unparseable strings that cut off mid-record due to generation token limits. |
| 6 | **Amount Stored as Raw Rupees Instead of Paise** | 25 | 1.7% | `fix` | Multiplied raw rupee/k/lakh amounts by 100 to restore true integer paise values. |
| 7 | **Unnormalized Transaction ID Format** | 20 | 1.4% | `fix` | Stripped hyphens and whitespace, uppercased to canonical `NP\d{10}`. |
| 8 | **Empty or Placeholder User Tickets** | 10 | 0.7% | `drop` | Removed synthetic artifact rows containing empty text or placeholder strings (`test`, `asdf`). |

---

## 3. Strict Cascading Order of Operations

To prevent compounding label errors, the remediation pipeline enforces a strict causal execution sequence on every ticket:

1. **Step 1: Normalizations & Conversions First**
   - Correct `amount_paise` (rupees to paise conversion errors, resolving `k`, `lakh`, and decimal figures).
   - Normalize `txn_id` to `NP\d{10}` (stripping hyphens and spaces).
   - Normalize `txn_date` from `DD/MM/YYYY` to `YYYY-MM-DD`.
   - Set non-transaction category transaction fields (`amount_paise`, `txn_date`, `txn_id`, `channel`) to `null`.
2. **Step 2: Priority Recomputation**
   - Recompute `priority` **after** `amount_paise` has been corrected. This is critical because tickets with amounts corrected from raw rupees (e.g., ₹46,094 stored as 46,094 paise) to true paise (4,609,400 paise) cross the ₹5,000 threshold and qualify for `P2`.
   - Apply legal threat keyword triggers (`RBI`, `ombudsman`, `consumer court`, `lawyer`, `legal notice`) and fraud category precedence.
3. **Step 3: Human Escalation Recomputation**
   - Recompute `needs_human` **after** `priority` has been finalized.
   - If `priority == "P1"`, `needs_human` is strictly enforced as `true`.
   - If `priority != "P1"`, `needs_human` is `true` if and only if explicit repeat-contact phrases exist in the ticket message (`"third time"`, `"again and again"`, `"baar baar"`, `"already contacted"`, `"pehle bhi"`, etc.); otherwise, set to `false`.
4. **Step 4: Canonical Key Ordering**
   - Reorder object keys to canonical order: `["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]`.

---

## 4. Verification of Subtle `SCHEMA.md` Traps

A targeted audit was conducted to verify edge cases and subtle rules on all remaining records:
- **Quoted Email Date Trap:** Verified that no ticket extracted its `txn_date` from quoted history lines (lines starting with `>`). In all 16 quoted email threads where a date was populated, the customer explicitly stated the date in the new message body.
- **Non-Transaction Metadata Trap:** Confirmed that all tickets with `category` in `["account_access", "kyc", "other"]` have `amount_paise: null`, `txn_date: null`, `txn_id: null`, and `channel: null`.
- **Language Label Trap:** Ran Hinglish token detection (`"hai"`, `"hoon"`, `"paise"`, `"kat gaye"`, etc.) across all remaining rows. Zero language label flips (`en` vs `hi-en`) were found among genuine tickets (the only 5 discrepancies in the raw data were on placeholder rows that were dropped).

---

## 5. Representative Sample Audit Log

| Row ID | Action | Reason |
|---|---|---|
| `tr-00020` | `fix` | Renamed 'prio' key to 'priority'; Normalized txn_date '02/09/2026' to '2026-09-02'; Corrected priority tier from 'low' to 'P3' per SCHEMA.md rules |
| `tr-00034` | `drop` | Corrupt empty or placeholder user ticket |
| `tr-00060` | `drop` | Truncated assistant response / unparseable JSON |
| `tr-00071` | `fix` | Normalized txn_id 'NP-9718716109' to 'NP9718716109' |
| `tr-00073` | `fix` | Corrected raw rupee amount_paise from 168787 to 16878725; Normalized txn_id 'NP 4271342441' to 'NP4271342441' |
| `tr-00156` | `drop` | Duplicate ticket text (identical to tr-00030) |
| `tr-00184` | `fix` | Corrected priority tier from 'P3' to 'P1' per SCHEMA.md rules |
| `tr-00208` | `drop` | Duplicate ticket text (identical to tr-00150) |
| `tr-00382` | `fix` | Corrected raw rupee amount_paise from 46094 to 4609400; Normalized txn_id 'np 33930 78402' to 'NP3393078402' |
| `tr-00727` | `fix` | Corrected raw rupee amount_paise from 17914 to 1791499 |
| `tr-00790` | `drop` | Corrupt empty or placeholder user ticket |
| `tr-01319` | `fix` | Corrected priority tier from 'P3' to 'P1' per SCHEMA.md rules |
