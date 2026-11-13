# Take-Home Assignment: Fine-Tune a Small Model for Ticket Triage

**Role:** Machine Learning Intern – LLM Fine-Tuning, VIMA3YA
**Expected effort:** one focused day (6–8 hours)
**Deadline:** 48 hours after you receive this pack
**Hardware:** Google Colab, free tier, T4 GPU only

## The task

NimbusPay (a fictional payments app) wants a small model that reads a support ticket and returns a triage record as JSON. The fields and the rules for filling them are in [SCHEMA.md](SCHEMA.md). Several fields need the model to apply business rules, convert formats and ignore distracting details.

At test time the model gets **only a short, fixed system prompt and the ticket**. It does not get the rules. Your job is to fine-tune a small open model so that it has learned the rules from the data.

## What you get

| File | What it is |
|---|---|
| `SCHEMA.md` | The labelling spec. Dev and test labels follow it exactly. |
| `data/train.jsonl` | 1,445 training tickets with labels. This data came out of a real-world-style pipeline and **it is not clean. Auditing it is part of the assignment.** |
| `data/dev.jsonl` | 200 clean, labelled tickets for you to measure progress. Do not train on these. |
| `data/test_inputs.jsonl` | 400 tickets without labels. You submit predictions for these. Part of this set is deliberately harder than dev. |
| `score.py` | The scorer we use. Run it on dev as often as you like. |

Each row is `{"id": ..., "messages": [system, user, assistant]}`. The assistant message is the label.

## Rules

**Tools.** Use any open-source training stack you like: Unsloth, Hugging Face PEFT/TRL, Axolotl, plain PyTorch. Tell us what you chose and why.

**Model and training**
- Any open-weights model with **at most 4B parameters**.
- **LoRA or QLoRA only**, with at most **50M trainable parameters**. No full fine-tuning.
- Everything must run on a free Colab T4.
- Save the final adapter in standard **PEFT format** (`adapter_config.json` + adapter weights) so we can load it.

**Inference (this is how we will run your model)**
- Input: exactly the `system` and `user` messages from the data, formatted with the model's own chat template. If your model has no system role, decide how to handle that and tell us.
- Plain `model.generate` with greedy decoding, `max_new_tokens = 200`.
- The output must be only the JSON object. The only post-processing allowed is stripping whitespace.
- **Not allowed:** putting the rules or examples in the prompt, constrained or grammar-based decoding, any code that fixes or fills fields after generation, and using another model or API to produce labels or predictions.

**Data**
- You may clean, fix, drop or augment the training data in any way that uses only `train.jsonl` and `SCHEMA.md`.
- Never train on dev or test rows.

**AI assistants.** You may use them, as you would on the job. Say where you did. You are responsible for every line and every number, and you will defend them live.

## What to do

### Part A: Audit the data (25 points)
Find out what is wrong with `train.jsonl` and decide what to do about each problem. Show how you found each one. Record every row you dropped or fixed in `cleaning_log.csv`.

### Part B: Baselines (10 points)
Measure on dev, with `score.py`:
1. your base model with the fixed prompt, no fine-tuning;
2. the same model with the rules written into the prompt (allowed here only, as a reference point).

### Part C: Fine-tune (20 points)
Build the training pipeline. We look for a correct chat template, loss on the right tokens, a sequence length chosen from evidence, and a clear reason for each main setting.

### Part D: Controlled experiments (20 points)
Run **at least four** experiments, changing one thing at a time:
1. raw training data vs your cleaned data;
2. 16-bit LoRA vs 4-bit QLoRA;
3. one of: rank, learning rate, number of epochs;
4. one idea of your own.

Report each in one table: what changed, dev exact match, dev mean field accuracy, peak GPU memory, training minutes.

### Part E: Error analysis (15 points)
Using your best model on dev: which fields and categories are weakest, and why? Show at least 10 specific failures with your explanation of each. Say what you would do next with one more week.

### Hidden test (10 points)
Scored by us from your `predictions_test.jsonl`.

## What to submit

Submit **one link to a GitHub repository**.

1. Create a **private** repository and push the files below to it while you work. Do not push the training data or the base model.
2. Keep it **private until the deadline**, so other candidates cannot see your work.
3. **Within 1 hour after the deadline**, make it public (Settings → General → Danger Zone → Change visibility) and reply to the assignment email with the link.

We review the last commit made before the deadline. Do not push anything after the deadline.

| File in your repository | Notes |
|---|---|
| `README.md` | Base model id, whether it was loaded in 4-bit, and a one-line summary of your best result. |
| `adapter/` | Your final LoRA adapter in PEFT format (`adapter_config.json` + `adapter_model.safetensors`). GitHub rejects files over 100 MB: if yours is larger, use Git LFS. Do not publish it anywhere public before the deadline. |
| `notebook.ipynb` | Your Colab notebook **with outputs**. It must run top to bottom on a fresh T4 runtime in under 3 hours and reproduce your final model and test predictions. Other experiments may be shown from saved logs, but include their code. |
| `report.md` or `report.pdf` | At most 3 pages: audit findings, experiment table, error analysis, what you would do next, and which AI tools you used for what. |
| `cleaning_log.csv` | Columns: `id,action,reason`. `action` is `drop` or `fix`. |
| `predictions_test.jsonl` | One line per test row: `{"id": "ts-00001", "output": "<raw model output>"}` |

## How we evaluate

- **Reproducibility check.** We load your adapter and regenerate a random sample of your test predictions. They must match your file.
- **Live defense (30 minutes).** You walk us through your notebook, explain your decisions, and make a small change live. This counts as much as the written submission.
- We care more about sound reasoning and honest reporting than about the top score. A clearly explained negative result is worth more than an unexplained good number.

## Practical tips

- Free Colab sessions disconnect and have daily GPU limits. Save adapters, logs and predictions to Google Drive as you go.
- Keep each training run short enough that four or more runs fit in your GPU quota.
- Look at the data before you train on it.

## Credits

Some ticket text is adapted from the Banking77 dataset (PolyAI, CC-BY-4.0). The task, the categories and all labels here are different from Banking77's.
