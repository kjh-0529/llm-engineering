"""QLoRA fine-tuning of Qwen3-1.7B on the Korean technical Q&A data.

    python train_qlora.py                       # T4: ~3.7k examples, 2 epochs
    python eval/evaluate.py --adapter runs/qlora --data data/test.jsonl --out results/qlora_test

The base model is loaded in 4-bit (NF4) and frozen; only LoRA adapters on every
attention and MLP projection are trained, with float16 compute (the T4 has no
bfloat16). Prompts use the same chat format and system prompt as evaluation,
and the loss is computed on answer tokens only. Without a GPU the script loads
the model unquantized in float32, which is only meant for a quick smoke test.
"""
import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))
from evaluate import build_prompt  # noqa: E402  (same prompt format as evaluation)


def load(path, tok, max_len, limit):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:limit]
    out = []
    for r in rows:
        # Tokenize prompt and answer separately so the loss mask boundary is exact.
        p = tok(build_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
        a = tok(r["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p + a)[:max_len]
        out.append({"input_ids": ids, "labels": ([-100] * len(p) + a)[:max_len]})
    return out


def collate(batch, pad_id):
    n = max(len(x["input_ids"]) for x in batch)
    pad = lambda x, v: x + [v] * (n - len(x))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in batch]),
        "labels": torch.tensor([pad(x["labels"], -100) for x in batch]),
        "attention_mask": torch.tensor([pad([1] * len(x["input_ids"]), 0) for x in batch]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "qlora"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N examples")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.eos_token == "<|im_end|>", tok.eos_token  # answers must end like chat turns
    if cuda:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                                     dtype=torch.float16, device_map="cuda")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train = load(os.path.join(HERE, "data", "train.jsonl"), tok, args.max_len, args.limit)
    val = load(os.path.join(HERE, "data", "val.jsonl"), tok, args.max_len, args.limit)
    # Self-check: the unmasked labels of the first example are exactly its answer.
    first = json.loads(open(os.path.join(HERE, "data", "train.jsonl"), encoding="utf-8").readline())
    target = tok.decode([t for t in train[0]["labels"] if t != -100], skip_special_tokens=True)
    assert target.strip() == first["answer"][: len(target.strip())], "loss mask misaligned"

    # 3% warm-up, computed here: newer transformers dropped `warmup_ratio`.
    steps = args.max_steps if args.max_steps > 0 else int(args.epochs * -(-len(train) // 16))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
            # 2 x 8 = 16 sequences per update. Batch 4 ran out of memory on the T4: the loss
            # materializes batch x 1,024 tokens x 152k-word vocabulary logits (4 GiB extra).
            per_device_train_batch_size=2, per_device_eval_batch_size=2,
            gradient_accumulation_steps=8, learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_steps=max(1, int(0.03 * steps)), logging_steps=10, eval_strategy="steps", eval_steps=100,
            save_strategy="no", fp16=cuda, gradient_checkpointing=cuda,
            optim="paged_adamw_8bit" if cuda else "adamw_torch",
            report_to="none", remove_unused_columns=False),
        train_dataset=train, eval_dataset=val,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)  # adapter only (~tens of MB)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
