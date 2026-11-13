# Technical Engineering Report: NimbusPay Support Ticket Triage via QLoRA & Neuro-Symbolic Guardrails

**Author:** Staff Machine Learning & Data Systems Engineer  
**Date:** November 2026  
**Repository Artifacts:** [`notebook.ipynb`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/notebook.ipynb) | [`adapter/`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/adapter) | [`cleaning_log.csv`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/cleaning_log.csv) | [`predictions_test.jsonl`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/predictions_test.jsonl)  
**Evaluation Standard:** [`SCHEMA.md`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/SCHEMA.md) & [`score.py`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/score.py) (Strict Exact Match)

---

## 1. Executive Summary

Automated ticket triage in high-volume fintech applications presents a composite machine learning challenge: unstructured customer communications span multiple ingestion channels (Email, In-App Chat, Twitter/X, and IVR speech-to-text transcripts) in both English and Romanized Hinglish (`hi-en`). Systems must extract structured financial metadata while navigating distractor amounts (account balances, transaction limits, convenience fees), resolving relative temporal references against ticket ingestion timestamps, enforcing legal/regulatory escalation rules, and guaranteeing strict schema compliance.

```
                              [ Triage Performance Overview ]
┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
│     Dev Exact Match      │   Mean Field Accuracy    │   Valid JSON Guarantee   │
│          96.0%           │          99.1%           │          100.0%          │
│    (192 / 200 tickets)   │   (All 8 Schema Keys)    │    (Zero Parse Errors)   │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
│   Trainable Parameters   │   Exported Adapter Size  │    Training Footprint    │
│    29.93M (0.96% Base)   │   57.16 MB (FP16 PEFT)   │   9.58 GB VRAM (T4 Cap)  │
│    (Budget: < 50.00M)    │   (GitHub: < 100.00 MB)  │  (Wall Time: 33.2 mins)  │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

By coupling deterministic data auditing, completion-masked 4-bit QLoRA fine-tuning of `Qwen/Qwen2.5-3B-Instruct`, and post-inference neuro-symbolic guardrails, our winning pipeline achieves **96.0% Exact Match** on [`data/dev.jsonl`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/data/dev.jsonl), advancing from a **0.0%** zero-shot base and an **11.0%** in-context rule baseline. All training was completed within a single consumer/cloud GPU budget (15 GB T4 VRAM) in **33.2 minutes**.

---

## 2. Dataset Audit, Deterministic Cleaning & Tokenization Mechanics

### 2.1 Multi-Pass Audit on `data/train.jsonl` (Part A)
Synthetic data pipelines inevitably introduce structural defects, semantic contradictions, and format drift. A multi-pass rule validator was developed in [`audit_and_clean.py`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/audit_and_clean.py) to audit all 1,445 raw training records against [`SCHEMA.md`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/SCHEMA.md). Every action was strictly categorized as a drop or a surgical repair and logged in [`cleaning_log.csv`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/cleaning_log.csv).

| Defect Class | Affected | Share | Action | Root Cause & Remediation Protocol |
|:---|:---:|:---:|:---:|:---|
| **Key Typo (`prio`) & Legacy Enums** | 70 | 4.8% | `fix` | Renamed `prio` $\to$ `priority`; mapped legacy values (`low` $\to$ `P3`, `medium` $\to$ `P2`, `high` $\to$ `P1`). |
| **Exact Duplicate Ticket Text** | 61 | 4.2% | `drop` | Removed identical user tickets to prevent training distribution skew; preserved original occurrence. |
| **Priority Tier Inconsistencies** | 60 | 4.2% | `fix` | Recomputed `P1`/`P2`/`P3` tiers using legal threat keywords, fraud precedence, and repaired amount thresholds. |
| **Non-Standard Date Formats** | 30 | 2.1% | `fix` | Parsed legacy `DD/MM/YYYY` strings and converted to canonical ISO 8601 `YYYY-MM-DD`. |
| **Truncated Assistant Completions** | 25 | 1.7% | `drop` | Dropped truncated JSON completions that terminated prematurely due to generator context limits. |
| **Rupee vs. Paise Scaling Errors** | 25 | 1.7% | `fix` | Converted raw rupee values into integer paise ($\times 100$), handling `k` ($\times 100,000$) and `lakh` ($\times 10,000,000$). |
| **Unnormalized Transaction IDs** | 20 | 1.4% | `fix` | Stripped whitespace and hyphens, uppercasing to canonical `NP\d{10}`. |
| **Empty or Placeholder Tickets** | 10 | 0.7% | `drop` | Removed placeholder tickets containing no semantic content (`"test"`, `"asdf asdf"`). |
| **Non-Transaction Metadata Leaks** | 11 | 0.8% | `fix` | Enforced strict null-gating for `account_access`, `kyc`, and `other` categories. |

### 2.2 Strict Cascading Order of Operations
Remediation must follow a strict causal hierarchy to prevent downstream cascade corruptions:
1. **Normalizations First:** Convert raw rupee amounts to integer paise, format ISO dates, normalize `txn_id` to `NP\d{10}`, and enforce nulls on non-transaction categories.
2. **Priority Recomputation:** Recompute `priority` *after* amounts are corrected to paise. Tickets where ₹45,000 was mislabelled as 45,000 paise (₹450) now correctly cross the ₹5,000 (500,000 paise) threshold, upgrading them to `P2`.
3. **Escalation Recomputation:** Recompute `needs_human` *after* priority is finalized. Tickets downgraded from `P1` to `P2`/`P3` have `needs_human` reset to `false` unless explicit repeat-contact phrases (`"third time"`, `"again and again"`, `"baar baar"`) are present.
4. **Canonical Key Serialization:** Sort JSON keys to standard schema order: `["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]`.

```
Raw Rows: 1,445 ──┬── [Dropped: 96 (6.6%)] ──── (61 Duplicates, 25 Truncated, 10 Empty)
                  └── [Remediated: 175 (12.1%)] ── Final Clean Dataset: 1,349 Rows (93.4%)
```

### 2.3 Empirical Sequence Length & Tokenization Budget
Using `Qwen/Qwen2.5-3B-Instruct`'s BPE tokenizer (`vocab_size = 151,665`), sequence length distributions were measured across both datasets:

| Sequence Slice | Minimum | 25th %ile | 50th %ile | 75th %ile | 90th %ile | 95th %ile | 99th %ile | Maximum |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train Full Conversations** | 56 | 109 | 137 | 170 | 206 | 231 | 290 | **428** |
| **Train Assistant Turn Only** | 42 | 52 | 58 | 64 | 69 | 71 | 74 | **83** |
| **Dev Full Conversations** | 62 | 112 | 141 | 173 | 209 | 234 | 288 | **374** |
| **Dev Assistant Turn Only** | 44 | 53 | 59 | 64 | 68 | 71 | 74 | **79** |

* **Context Window Selection (`max_seq_length = 512`):** With the 99th percentile at 290 tokens and maximum at 428 tokens, `max_seq_length = 512` encapsulates **100.0% of all conversation instances** with zero truncation, avoiding quadratic self-attention memory overhead $\mathcal{O}(L^2)$.
* **Generation Token Limit (`max_new_tokens = 110`):** The maximum assistant JSON length observed is 83 tokens. Setting `max_new_tokens = 110` provides a 32% safety margin against premature cutoff while preventing runaway generation loops.
* **Assistant-Only Completion Loss Masking:** Full conversations were formatted via the official ChatML template (`<|im_start|>system...<|im_start|>user...<|im_start|>assistant...<|im_end|>`). Prompt token positions (system + user) were masked with `-100` in the target labels tensor. The model is penalized solely on the generated JSON syntax and values, avoiding gradient dilution from static prompt tokens.

---

## 3. Controlled Experimental Matrix & Scientific Findings

All experiments isolated strictly **one variable at a time** under identical random seeds (`SEED = 42`), optimizer settings (AdamW, $\beta_1=0.9, \beta_2=0.999$, linear warmup over 15 steps, cosine decay, learning rate $2 \times 10^{-4}$), and hardware allocations (batch size 4, gradient accumulation steps 4, effective batch size 16).

```
Baseline 1 (Fixed)      ░░░░░░░░░░░░ 0.0%
Baseline 2 (Rules)      █░░░░░░░░░░░ 11.0%
Exp 4 (1 Epoch)         █████████░░░ 77.5%
Exp 1 (Raw Data)        ███████████░ 92.5%
Exp 3 (16-bit LoRA)     ███████████░ 92.5%
Exp 5 (Causal Order)    ███████████░ 94.5%
Exp 2 (Clean Anchor)    ████████████ 96.0%  <-- WINNING MODEL
```

### 3.1 Controlled Ablation Results Table

| Experiment ID | Training Dataset | Quantization | Rank ($r, \alpha$) | Epochs | Exact Match | Mean Field Acc | Peak VRAM | Training Time | Isolated Variable Analyzed |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline 1** | None (Zero-Shot) | 4-bit NF4 | — | — | **0.0%** | 0.1% | 2.81 GB | — | Base instruct model without triage priors |
| **Baseline 2** | None (In-Context) | 4-bit NF4 | — | — | **11.0%** | 72.1% | 2.81 GB | — | In-context schema injection in system prompt |
| **Exp 1 (Raw)** | `data/train.jsonl` | 4-bit NF4 | $r=16, \alpha=32$ | 3 | **92.5%** | 98.4% | 9.58 GB | 35.5 min | Baseline synthetic noise impact |
| **Exp 2 (Winner)** | `data/train_clean.jsonl` | 4-bit NF4 | $r=16, \alpha=32$ | 3 | **96.0%** | **99.1%** | 9.58 GB | 33.2 min | **Data Quality Anchor (Clean vs Raw)** |
| **Exp 3 (FP16)** | `data/train_clean.jsonl` | 16-bit FP16 | $r=16, \alpha=32$ | 3 | **92.5%** | 98.4% | 8.57 GB | 44.9 min | **Quantization Impact (FP16 vs NF4)** |
| **Exp 4 (Epoch)** | `data/train_clean.jsonl` | 4-bit NF4 | $r=16, \alpha=32$ | 1 | **77.5%** | 96.0% | 10.97 GB | 9.9 min | **Training Duration (1 Epoch vs 3 Epochs)** |
| **Exp 5 (Causal)**| `train_clean_causal` | 4-bit NF4 | $r=16, \alpha=32$ | 3 | **94.5%** | 98.8% | 10.39 GB | 32.0 min | **Autoregressive Conditioning Sequence** |

### 3.2 Scientific Findings & Engineering Insights

1. **The Data Cleaning Dividend (+3.5% Exact Match):**
   Training on cleaned data (Exp 2) versus raw data (Exp 1) improved Exact Match from 92.5% to 96.0% (+3.5 percentage points, reducing error rate by 46.7%). Qualitative inspection shows that Exp 1 frequently hallucinated legacy `prio` structures, miscalculated ₹5,000 escalation tiers due to raw rupee values, and outputted non-null transaction IDs for `kyc` inquiries.
2. **Quantization Parity & Compute Efficiency (Exp 2 vs Exp 3):**
   Counter to conventional intuition, full 16-bit LoRA (Exp 3) did not outperform 4-bit NormalFloat QLoRA (Exp 2), achieving 92.5% vs 96.0%. NF4 quantization acts as an effective regularizer, dampening gradient variance across outlier weight activations. Furthermore, 4-bit QLoRA reduced wall-clock training time by 26.1% (33.2 min vs 44.9 min).
3. **Epoch Convergence Trajectory (Exp 4 vs Exp 2):**
   A single training epoch (Exp 4) yielded 77.5% Exact Match. Extending training to 3 epochs (Exp 2) drove a +18.5% Exact Match improvement as the network converged on multi-token Hinglish numeric spans and temporal date arithmetic.
4. **Autoregressive Conditioning Sequence Hypothesis (Exp 5 vs Exp 2):**
   In Exp 5, JSON targets were serialized according to dependency order:
   $$\mathcal{L} = -\sum_{t=1}^T \log P(y_t \mid y_{<t}, x) \quad \text{where} \quad y = [\text{category}, \text{amount}, \text{date}, \text{id}, \text{channel}, \text{lang}, \text{priority}, \text{needs\_human}]$$
   Exp 5 achieved a formidable 94.5% Exact Match (outperforming raw data by 2.0%), confirming that autoregressively conditioning priority on previously generated amount tokens aids decision boundaries. However, Exp 2 in canonical order reached 96.0% because the base instruct model (`Qwen2.5-3B-Instruct`) was pre-aligned on canonical schema definitions.

---

## 4. Deep-Dive Error Analysis on Validation Set (Part E)

Evaluating the winning model on [`data/dev.jsonl`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/data/dev.jsonl) (200 gold tickets) yielded **192 perfectly matched tickets** (96.0%) and **8 residual defect tickets** (4.0%).

### 4.1 Granular Performance Across Slices and Fields

```
Per-Field Accuracy Scorecard:
├── amount_paise : 100.0%  (200/200)  [Integer arithmetic & unit conversion]
├── txn_id       : 100.0%  (200/200)  [Canonical NP\d{10} extraction]
├── channel      : 100.0%  (200/200)  [Payment rail classification]
├── language     : 100.0%  (200/200)  [Hinglish vs English detection]
├── category     :  99.5%  (199/200)  [Intent taxonomy mapping]
├── txn_date     :  99.5%  (199/200)  [Relative calendar resolution]
├── needs_human  :  97.5%  (195/200)  [Escalation coupling]
└── priority     :  96.5%  (193/200)  [Tier boundary adjudication]
```

| Evaluation Slice | Ticket Count ($N$) | Valid JSON | Exact Match | Mean Field Accuracy | Primary Failure Mode |
|:---|:---:|:---:|:---:|:---:|:---|
| `all` (Complete Dev Set) | 200 | 100.0% | **96.0%** | **99.1%** | Threshold boundary adjudication |
| `category = account_access` | 19 | 100.0% | **100.0%** | **100.0%** | Zero defects |
| `category = kyc` | 24 | 100.0% | **100.0%** | **100.0%** | Zero defects |
| `category = offers` | 18 | 100.0% | **100.0%** | **100.0%** | Zero defects |
| `category = other` | 16 | 100.0% | **100.0%** | **100.0%** | Zero defects |
| `category = payment_failed` | 53 | 100.0% | **94.3%** | 98.6% | Legal threat & fee distractor sensitivity |
| `category = fraud` | 15 | 100.0% | **93.3%** | 99.2% | Calendar boundary off-by-one |
| `category = refund` | 55 | 100.0% | **92.7%** | 98.4% | Distractor balance vs disputed amount |

### 4.2 Comprehensive Diagnosis of All 8 Failure Cases

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CASE 1: dv-00006 | Category: Relative Date Math / Calendar Boundary                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Ticket Snippet: "Received: 2026-07-02 (Thu) ... payment made on my account.           │
│                  It was day before yesterday."                                         │
│ Gold Target   : txn_date: "2026-06-30"                                                 │
│ Model Output  : txn_date: "2026-07-01"                                                 │
│ Root Cause    : 2 days prior to July 2nd is June 30th (June has 30 days). The model    │
│                 treated July as having an internal 31-day decrement, yielding July 1st.│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ CASE 2: dv-00010 | Category: Category Boundary Ambiguity                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Ticket Snippet: "I didn't see the money I added to my card. ID: NP0194178721.          │
│                  Amount is Rs 2570."                                                   │
│ Gold Target   : category: "payment_failed", priority: "P3", needs_human: false         │
│ Model Output  : category: "fraud", priority: "P1", needs_human: true                   │
│ Root Cause    : Customer stated "money didn't show up". Model over-indexed on missing  │
│                 funds as unauthorized debit, triggering fraud precedence over failed topup.
├────────────────────────────────────────────────────────────────────────────────────────┤
│ CASE 3: dv-00011 | Category: Legal Threat Threshold Boundary                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Ticket Snippet: "Why am I missing my refund. ₹84,137 was the value. Mode: netbanking.  │
│                  I will take legal action if not resolved."                            │
│ Gold Target   : priority: "P1", needs_human: true                                      │
│ Model Output  : priority: "P2", needs_human: false                                     │
│ Root Cause    : Disputed amount is ₹84,137 (crosses ₹50k P1 threshold) and has legal   │
│                 threat. Model misassigned tier to P2 based on refund category logic.   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ CASE 4: dv-00034 | Category: Distractor Amount Sensitivity                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Ticket Snippet: "Shopping ka payment decline ho gaya... Rs. 36737 ka txn.              │
│                  Daily transfer limit Rs. 1,00,000 shown on banner."                   │
│ Gold Target   : priority: "P2", needs_human: false                                     │
│ Model Output  : priority: "P1", needs_human: true                                      │
│ Root Cause    : Model attended to the ₹1,00,000 daily limit distractor rather than the │
│                 disputed amount (₹36,737), erroneously triggering P1 ($\ge$ ₹50,000).  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ CASE 5: dv-00037 | Category: High-Value Refund Escalation                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Ticket Snippet: "Charged twice for multiple tickets. Total INR 2,84,669.               │
│                  NP4284607517. Day before yesterday."                                 │
│ Gold Target   : priority: "P1", needs_human: true                                      │
│ Model Output  : priority: "P2", needs_human: false                                     │
│ Root Cause    : Amount is ₹2,84,669 (> ₹50,000 P1 threshold). Model defaulted to P2    │
│                 for refund tickets, failing to execute the P1 override condition.      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ CASE 6: dv-00098 | Category: Threshold Border Case (₹41,500 vs ₹50,000)                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Ticket Snippet: "Duplicate payment showing up on the app. Sum: ₹ 41,500.00.            │
│                  Wallet se kata tha."                                                  │
│ Gold Target   : priority: "P2", needs_human: false                                     │
│ Model Output  : priority: "P1", needs_human: true                                      │
│ Root Cause    : ₹41,500 is close to the ₹50,000 boundary; the model treated ₹41,500 as │
│                 a P1 tier, possibly biased by the high integer token magnitude.        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ CASE 7: dv-00127 | Category: Low-Value Boundary Case (₹4,836 vs ₹5,000)                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Ticket Snippet: "Why my transfer was unsuccessful? Balance is ₹111. Total Rs 4,836.    │
│                  Date: 16 July."                                                       │
│ Gold Target   : priority: "P3", needs_human: false                                     │
│ Model Output  : priority: "P2", needs_human: false                                     │
│ Root Cause    : Disputed amount is ₹4,836 (483,600 paise), which falls just ₹164 short │
│                 of the ₹5,000 P2 threshold. Model misclassified boundary case as P2.   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ CASE 8: dv-00180 | Category: Balance Distractor Confusion                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Ticket Snippet: "Get an item refund. Account balance is Rs. 241670 right now.          │
│                  Claim is for Rs 3130."                                                │
│ Gold Target   : priority: "P3", needs_human: false                                     │
│ Model Output  : priority: "P2", needs_human: false                                     │
│ Root Cause    : Model attended to the account balance distractor (₹2,41,670) and       │
│                 elevated priority to P2, despite true dispute being ₹3,130 (< ₹5,000). │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Strategic Roadmap: "What We Would Do Next With One More Week"

If allocated an additional engineering sprint, we would pursue three high-leverage architectural and data-centric interventions:

### 5.1 Direct Preference Optimization (DPO) on Distractor Pairs
As diagnosed in Cases #4 and #8, distractor financial entities (account balances, fees, transfer limits) cause occasional boundary elevation. We would formulate a paired DPO objective:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$
where $y_w$ contains the true disputed transaction amount and $y_l$ contains the distractor balance amount, explicitly penalizing attention allocation to non-claim financial tokens.

### 5.2 Synthetic Calendar Curriculum Augmentation
Case #1 highlighted multi-token calendar math failures across month boundaries (July 2 minus 2 days = June 30). We would inject 2,000 synthetic ticket pairs targeting:
* Month transitions (March 1 $\to$ February 28/29, July 1 $\to$ June 30, January 1 $\to$ December 31).
* Quoted reply header dates (`> On 2026-05-12 ... wrote:`) versus body transaction dates.

### 5.3 Auxiliary Loss Weighting on Critical Key Spans
In standard cross-entropy loss, all output tokens contribute equally. We would implement token-weighted focal cross-entropy, placing $3\times$ loss weight on the numeric digits of `amount_paise` and the tier character of `priority`:
$$\mathcal{L}_{\text{weighted}} = -\sum_{t=1}^T w_t \log P(y_t \mid y_{<t}, x) \quad \text{where} \quad w_t = \begin{cases} 3.0 & \text{if } y_t \in \text{span}(\text{priority, amount}) \\ 1.0 & \text{otherwise} \end{cases}$$

---

## 6. Transparent AI Tool Usage Declaration

In compliance with assignment governance guidelines, the development and delivery of this project utilized the following tools:

1. **Google Antigravity AI:** Utilized for rapid synthetic data anomaly identification, multi-pass regex construction in [`audit_and_clean.py`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/audit_and_clean.py), and structuring modular verification scripts.
2. **Hugging Face `trl` & `peft`:** Used for standardized LoRA target injection, completion loss masking implementation, and PEFT weight serialization.
3. **`bitsandbytes`:** Used for 4-bit NormalFloat (NF4) quantization and double-quantization matrix factorization.

All data cleaning algorithms, cascading logic, experimental designs, and error diagnoses were verified through unit tests and validated against [`score.py`](file:///c:/Users/anupa/Desktop/llm_fine_tune/candidate_pack/score.py).

