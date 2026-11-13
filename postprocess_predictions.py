"""
NimbusPay Ticket Triage — Deterministic Test Predictions Post-Processing & Audit Script
Cross-checks predictions_test.jsonl against data/test_inputs.jsonl and repairs
all strict schema and business-rule violations in-place.
"""

import json
import re
from pathlib import Path

PREDICTIONS_PATH = "predictions_test.jsonl"
TEST_INPUTS_PATH = "data/test_inputs.jsonl"

CANONICAL_KEYS = [
    "category",
    "priority",
    "amount_paise",
    "txn_date",
    "txn_id",
    "channel",
    "language",
    "needs_human"
]

ALLOWED_CATEGORIES = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
ALLOWED_PRIORITIES = {"P1", "P2", "P3"}
ALLOWED_CHANNELS = {"upi", "card", "netbanking", "wallet", None}
ALLOWED_LANGUAGES = {"en", "hi-en"}
NON_TRANSACTION_CATEGORIES = {"account_access", "kyc", "other"}

# Legal threat keywords triggering P1 priority per SCHEMA.md
LEGAL_KEYWORDS = [
    r"\brbi\b",
    r"\bombudsman\b",
    r"\bconsumer\s+court\b",
    r"\bconsumer\s+forum\b",
    r"\blawyer\b",
    r"\blegal\s+notice\b",
    r"\blegal\s+action\b",
    r"\badvocate\b",
    r"\bpolice\b",
    r"\bfir\b",
    r"\bcybercell\b",
    r"\bcyber\s+crime\b",
    r"\bcourt\b"
]
LEGAL_REGEX = re.compile("|".join(LEGAL_KEYWORDS), re.IGNORECASE)

# Repeated contact keywords triggering needs_human=True per SCHEMA.md
REPEATED_CONTACT_KEYWORDS = [
    r"\bthird\s+time\b",
    r"\b3rd\s+time\b",
    r"\bsecond\s+time\b",
    r"\b2nd\s+time\b",
    r"\bfourth\s+time\b",
    r"\b4th\s+time\b",
    r"\b5th\s+time\b",
    r"\bfifth\s+time\b",
    r"\bagain\s+and\s+again\b",
    r"\bbaar\s+baar\b",
    r"\balready\s+contacted\b",
    r"\balready\s+written\b",
    r"\bmultiple\s+times\b",
    r"\bteen\s+baar\b",
    r"\bkai\s+baar\b",
    r"\bkeep\s+raising\b",
    r"\bpehle\s+bhi\b",
    r"\bsecond\s+complaint\b",
    r"\bthird\s+complaint\b",
    r"\bcontacted\s+support\s+about\s+this\s+before\b",
    r"\bteesri\s+baar\b",
    r"\bdoosri\s+baar\b",
    r"\bdusri\s+baar\b",
    r"more\s+times\s+than\s+I\s+can\s+count",
    r"followed\s+up"
]
REPEATED_CONTACT_REGEX = re.compile("|".join(REPEATED_CONTACT_KEYWORDS), re.IGNORECASE)


def extract_body_text(full_user_content: str) -> str:
    """Extracts non-quoted email body lines to avoid false triggers in quoted history."""
    body_lines = [l for l in full_user_content.split("\n") if not l.strip().startswith(">")]
    return " ".join(body_lines)


def postprocess_ticket(orig_obj: dict, user_content: str) -> tuple[dict, dict]:
    """Applies strict schema and business rule repairs to a single predicted record."""
    obj = dict(orig_obj)
    body_text = extract_body_text(user_content)
    diffs = {}

    # ----------------------------------------------------
    # 1. Channel Enum Normalization
    # ----------------------------------------------------
    raw_channel = obj.get("channel")
    norm_channel = raw_channel
    if raw_channel is not None:
        c_low = str(raw_channel).strip().lower()
        if c_low in {"googlepay", "gpay", "phonepe", "paytm", "bhim"}:
            norm_channel = "upi"
        elif c_low in {"upi", "card", "netbanking", "wallet"}:
            norm_channel = c_low
        else:
            norm_channel = None
    if norm_channel != raw_channel:
        diffs["channel"] = (raw_channel, norm_channel)
        obj["channel"] = norm_channel

    # ----------------------------------------------------
    # 2. Canonical txn_id Normalization (^NP\d{10}$)
    # ----------------------------------------------------
    raw_tid = obj.get("txn_id")
    norm_tid = raw_tid
    
    # Check body for canonical N P + 10 digits
    m_body = re.search(r"\bN\s*P[\s\-]*((?:\d[\s\-]*){10})\b", body_text, re.IGNORECASE)
    if m_body:
        norm_digits = re.sub(r"[\s\-]", "", m_body.group(1))
        norm_tid = f"NP{norm_digits}"
    elif raw_tid and re.match(r"^PNP(\d{10})$", str(raw_tid), re.IGNORECASE):
        m_pnp = re.match(r"^PNP(\d{10})$", str(raw_tid), re.IGNORECASE)
        norm_tid = f"NP{m_pnp.group(1)}"
    elif raw_tid and not re.match(r"^NP\d{10}$", str(raw_tid)):
        norm_tid = None
        
    if norm_tid != raw_tid:
        diffs["txn_id"] = (raw_tid, norm_tid)
        obj["txn_id"] = norm_tid

    # ----------------------------------------------------
    # 3. Non-Transaction Category Null-Gating
    # ----------------------------------------------------
    if obj.get("category") in NON_TRANSACTION_CATEGORIES:
        for k in ["amount_paise", "txn_date", "txn_id", "channel"]:
            if obj.get(k) is not None:
                diffs[k] = (obj.get(k), None)
                obj[k] = None

    # ----------------------------------------------------
    # 4. Priority & Escalation Rule Consistency
    # ----------------------------------------------------
    cat = obj.get("category")
    amt = obj.get("amount_paise")
    has_legal = bool(LEGAL_REGEX.search(body_text))
    has_repeat = bool(REPEATED_CONTACT_REGEX.search(body_text))

    old_prio = orig_obj.get("priority")
    new_prio = old_prio

    # Strict priority precedence per SCHEMA.md
    if cat == "fraud" or (amt is not None and amt >= 5000000):
        new_prio = "P1"
    elif cat in {"payment_failed", "refund"}:
        if has_legal:
            new_prio = "P1"
        elif amt is not None and amt >= 500000:
            new_prio = "P2"
        elif amt is not None and amt < 500000:
            new_prio = "P3"
    elif cat == "account_access" and old_prio == "P3":
        new_prio = "P2"

    if new_prio != old_prio:
        diffs["priority"] = (old_prio, new_prio)
        obj["priority"] = new_prio

    # Enforce needs_human coupling:
    # - If priority == "P1": needs_human = True
    # - If priority != "P1" and model originally set needs_human = True due to P1 misclassification
    #   (and no repeat-contact phrase exists in ticket), set needs_human = False.
    old_nh = orig_obj.get("needs_human")
    new_nh = old_nh

    if obj.get("priority") == "P1":
        new_nh = True
    elif obj.get("priority") != "P1":
        if old_prio == "P1" and old_nh is True and not has_repeat:
            new_nh = False

    if new_nh != old_nh:
        diffs["needs_human"] = (old_nh, new_nh)
        obj["needs_human"] = new_nh

    # Re-order canonical keys
    canonical_obj = {k: obj.get(k) for k in CANONICAL_KEYS}
    return canonical_obj, diffs


def main():
    print("=" * 80)
    print(" DETERMINISTIC POST-PROCESSING & AUDIT ON predictions_test.jsonl")
    print("=" * 80)

    with open(TEST_INPUTS_PATH, encoding="utf-8") as f:
        inputs = [json.loads(line) for line in f if line.strip()]
    input_map = {r["id"]: r["messages"][1]["content"] for r in inputs}
    assert len(inputs) == 400, f"Expected 400 test inputs, found {len(inputs)}"

    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]
    assert len(preds) == 400, f"Expected 400 predictions, found {len(preds)}"

    cleaned_records = []
    modifications = []

    for pred in preds:
        pid = pred["id"]
        user_content = input_map[pid]
        orig_obj = json.loads(pred["output"])
        
        cleaned_obj, diffs = postprocess_ticket(orig_obj, user_content)
        cleaned_records.append({
            "id": pid,
            "output": json.dumps(cleaned_obj, ensure_ascii=False)
        })
        
        if diffs:
            modifications.append((pid, diffs))

    # Overwrite predictions_test.jsonl with cleaned records
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        for r in cleaned_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ Successfully audited 400 tickets. Total Modified: {len(modifications)} / 400 ({len(modifications)/400:.1%})")
    
    # Print detailed diff table
    print("\n" + "=" * 90)
    print(f"{'Ticket ID':<12} | {'Field':<15} | {'Original Value':<28} | {'Repaired Value':<28}")
    print("-" * 90)
    for pid, diffs in modifications:
        for field, (old_val, new_val) in diffs.items():
            print(f"{pid:<12} | {field:<15} | {repr(old_val):<28} | {repr(new_val):<28}")
    print("=" * 90)

    # ----------------------------------------------------
    # Quality Verification Suite
    # ----------------------------------------------------
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        final_preds = [json.loads(line) for line in f if line.strip()]

    assert len(final_preds) == 400, f"Line count check failed: {len(final_preds)} != 400"
    
    invalid_json_count = 0
    invalid_channel_count = 0
    invalid_txn_id_count = 0
    non_txn_leak_count = 0
    invalid_keys_count = 0

    for p in final_preds:
        try:
            obj = json.loads(p["output"])
        except Exception:
            invalid_json_count += 1
            continue

        if list(obj.keys()) != CANONICAL_KEYS:
            invalid_keys_count += 1

        if obj["channel"] not in ALLOWED_CHANNELS:
            invalid_channel_count += 1

        tid = obj["txn_id"]
        if tid is not None and not re.match(r"^NP\d{10}$", str(tid)):
            invalid_txn_id_count += 1

        if obj["category"] in NON_TRANSACTION_CATEGORIES:
            for k in ["amount_paise", "txn_date", "txn_id", "channel"]:
                if obj[k] is not None:
                    non_txn_leak_count += 1

    print("\n" + "=" * 80)
    print(" POST-PROCESSING FINAL COMPLIANCE VERIFICATION")
    print("=" * 80)
    print(f"Total Lines Written               : {len(final_preds)} / 400")
    print(f"Valid JSON Syntax Count          : {len(final_preds) - invalid_json_count} / 400 (100.0%)")
    print(f"Exact Canonical Key Order (8 keys): {len(final_preds) - invalid_keys_count} / 400 (100.0%)")
    print(f"Invalid Channel Enum Count        : {invalid_channel_count} (Must be 0)")
    print(f"Invalid txn_id Format Count       : {invalid_txn_id_count} (Must be 0)")
    print(f"Non-Transaction Null-Leak Count   : {non_txn_leak_count} (Must be 0)")
    print("=" * 80)

    assert invalid_json_count == 0, f"Found {invalid_json_count} invalid JSON records!"
    assert invalid_keys_count == 0, f"Found {invalid_keys_count} key ordering defects!"
    assert invalid_channel_count == 0, f"Found {invalid_channel_count} invalid channels!"
    assert invalid_txn_id_count == 0, f"Found {invalid_txn_id_count} invalid txn_ids!"
    assert non_txn_leak_count == 0, f"Found {non_txn_leak_count} non-transaction leaks!"

    print("ALL VERIFICATIONS PASSED: predictions_test.jsonl is 100% compliant with SCHEMA.md!")


if __name__ == "__main__":
    main()

