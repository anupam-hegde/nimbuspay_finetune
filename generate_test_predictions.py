"""
NimbusPay Ticket Triage — Test Predictions Generator & Validation Pipeline
Generates predictions_test.jsonl from data/test_inputs.jsonl using local GPU.
"""

import os
import gc
import json
import time
from pathlib import Path
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./adapter"
TEST_INPUTS_PATH = "data/test_inputs.jsonl"
OUTPUT_PREDICTIONS_PATH = "predictions_test.jsonl"
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

def check_environment():
    print("=" * 80)
    print(" 1. HARDWARE & ENVIRONMENT VERIFICATION")
    print("=" * 80)
    assert torch.cuda.is_available(), "CUDA is NOT available! Local GPU is required."
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ PyTorch Version : {torch.__version__}")
    print(f"✓ CUDA Device     : {device_name}")
    print(f"✓ Total VRAM      : {vram_gb:.2f} GB")
    torch.cuda.empty_cache()


def ensure_adapter_float16(adapter_dir: str):
    print("\n" + "=" * 80)
    print(" 2. GITHUB FILE SIZE COMPLIANCE (< 100 MB)")
    print("=" * 80)
    weights_path = Path(adapter_dir) / "adapter_model.safetensors"
    assert weights_path.exists(), f"Adapter weights not found at {weights_path}"
    
    current_size_mb = weights_path.stat().st_size / (1024**2)
    print(f"Current adapter file size: {current_size_mb:.2f} MB")
    
    if current_size_mb >= 100.0:
        print(">> File size >= 100 MB. Converting float32 weights to float16 in-place...")
        state_dict = load_file(str(weights_path))
        converted = {}
        for k, v in state_dict.items():
            if v.dtype == torch.float32:
                converted[k] = v.to(torch.float16)
            else:
                converted[k] = v
        save_file(converted, str(weights_path))
        new_size_mb = weights_path.stat().st_size / (1024**2)
        print(f"✓ Converted to float16. New file size: {new_size_mb:.2f} MB (< 100 MB GitHub limit)")
        assert new_size_mb < 100.0, f"Size after conversion ({new_size_mb:.2f} MB) still >= 100 MB!"
    else:
        print(f"✓ Adapter weights file size ({current_size_mb:.2f} MB) is already < 100 MB.")
    
    return weights_path.stat().st_size / (1024**2)


def load_model_and_tokenizer(model_id: str, adapter_dir: str):
    print("\n" + "=" * 80)
    print(" 3. MODEL & TOKENIZER INITIALIZATION")
    print("=" * 80)
    print(f"Loading tokenizer from {adapter_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    if not tokenizer.chat_template:
        jinja_path = Path(adapter_dir) / "chat_template.jinja"
        if jinja_path.exists():
            tokenizer.chat_template = jinja_path.read_text(encoding="utf-8")
            
    print(f"✓ Tokenizer ready: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}, padding_side={tokenizer.padding_side}")

    print(f"Loading base model '{model_id}' in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    print("✓ Base model loaded in 4-bit NF4 precision.")

    print(f"Attaching LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    print("✓ PEFT adapter attached and configured in eval mode with use_cache=True.")

    return model, tokenizer


def generate_predictions(model, tokenizer, test_rows, batch_size=2, max_new_tokens=110):
    print("\n" + "=" * 80)
    print(f" 4. DETERMINISTIC INFERENCE ON {len(test_rows)} TEST TICKETS")
    print(f"    Batch Size: {batch_size} | max_new_tokens: {max_new_tokens} | Greedy (do_sample=False)")
    print("=" * 80)
    
    predictions = []
    start_time = time.time()
    
    for i in tqdm(range(0, len(test_rows), batch_size), desc="Inferring"):
        batch_rows = test_rows[i : i + batch_size]
        batch_prompts = [
            tokenizer.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True)
            for r in batch_rows
        ]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        for idx, row in enumerate(batch_rows):
            gen_ids = output_ids[idx][prompt_len:]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            predictions.append({"id": row["id"], "output": raw_text})
            
    elapsed = time.time() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"✓ Inference complete in {elapsed:.1f}s ({len(test_rows)/elapsed:.2f} tickets/sec)")
    print(f"✓ Peak VRAM allocated: {peak_vram_gb:.2f} GB")
    
    return predictions


def validate_and_save_predictions(predictions, test_rows, output_path):
    print("\n" + "=" * 80)
    print(" 5. AUTOMATED SCHEMA & QUALITY VALIDATION")
    print("=" * 80)
    
    assert len(predictions) == len(test_rows), f"Count mismatch: {len(predictions)} vs {len(test_rows)}"
    
    expected_ids = [r["id"] for r in test_rows]
    pred_ids = [p["id"] for p in predictions]
    assert pred_ids == expected_ids, "Prediction IDs do not match input IDs in exact order!"
    
    schema_errors = 0
    valid_json_count = 0
    exact_keys_count = 0
    
    for p in predictions:
        out_str = p["output"]
        try:
            obj = json.loads(out_str)
            valid_json_count += 1
            if set(obj.keys()) == set(CANONICAL_KEYS):
                exact_keys_count += 1
            else:
                schema_errors += 1
        except Exception:
            schema_errors += 1
            
    print(f"Total Predictions Evaluated : {len(predictions)}")
    print(f"Valid JSON Syntactically   : {valid_json_count} / {len(predictions)} ({valid_json_count/len(predictions):.1%})")
    print(f"Exact 8 Schema Keys Present : {exact_keys_count} / {len(predictions)} ({exact_keys_count/len(predictions):.1%})")
    print(f"Total Schema Errors         : {schema_errors}")
    assert schema_errors == 0, f"Encountered {schema_errors} schema errors in predictions!"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    print(f"✓ Successfully wrote {len(predictions)} verified predictions to {output_path}")
    return schema_errors


def verify_determinism(model, tokenizer, test_rows, predictions, num_samples=4):
    print("\n" + "=" * 80)
    print(f" 6. DETERMINISM AUDIT ON FIRST {num_samples} SAMPLES")
    print("=" * 80)
    
    sample_rows = test_rows[:num_samples]
    sample_preds = generate_predictions(model, tokenizer, sample_rows, batch_size=2, max_new_tokens=110)
    
    for i in range(num_samples):
        orig = predictions[i]["output"]
        regen = sample_preds[i]["output"]
        assert orig == regen, f"Non-deterministic generation detected on row {sample_rows[i]['id']}!"
        print(f"✓ Ticket {sample_rows[i]['id']}: Bit-for-bit identical output confirmed.")
        
    print("✓ Determinism verification passed! 100% bit-for-bit reproducible.")
    return True


def main():
    check_environment()
    adapter_size_mb = ensure_adapter_float16(ADAPTER_DIR)
    
    with open(TEST_INPUTS_PATH, encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f if line.strip()]
    assert len(test_rows) == 400, f"Expected 400 test rows, found {len(test_rows)}"
    
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, ADAPTER_DIR)
    predictions = generate_predictions(model, tokenizer, test_rows, batch_size=2, max_new_tokens=110)
    schema_errors = validate_and_save_predictions(predictions, test_rows, OUTPUT_PREDICTIONS_PATH)
    determinism_ok = verify_determinism(model, tokenizer, test_rows, predictions, num_samples=4)
    
    print("\n" + "=" * 80)
    print(" FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"✓ Adapter Size (MB)       : {adapter_size_mb:.2f} MB (< 100 MB GitHub limit)")
    print(f"✓ Total Test Predictions  : {len(predictions)} / 400")
    print(f"✓ Schema Error Count      : {schema_errors}")
    print(f"✓ Determinism Verified    : {determinism_ok}")
    print(f"✓ Output File Path        : {OUTPUT_PREDICTIONS_PATH}")
    print("=" * 80)
    print("ALL REQUIREMENTS SUCCESSFULLY COMPLETED!")


if __name__ == "__main__":
    main()
