"""Generate answers with a base model (optionally + a LoRA adapter) and score them.

    python eval/evaluate.py --model Qwen/Qwen3-1.7B --data data/test.jsonl \
        --out results/base_test
    python eval/evaluate.py --model Qwen/Qwen3-1.7B --adapter runs/qlora \
        --data data/test.jsonl --out results/qlora_test

Greedy decoding, the same system prompt for every model, Qwen3 "thinking" mode
off. Writes <out>.jsonl (every answer with its scores) and <out>.summary.json.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score, summarize  # noqa: E402

SYSTEM_PROMPT = "당신은 정확하고 간결하게 한국어로 답하는 기술 도우미입니다."


def build_prompt(tok, question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none",
                    help="bitsandbytes weight quantization (GPU only)")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    # T4 has no bfloat16: use float16 on GPU, float32 on CPU (smoke tests only).
    dtype = torch.float16 if use_cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # decoder-only batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = None
    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        quant = (BitsAndBytesConfig(load_in_8bit=True) if args.quant == "8bit" else
                 BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, quantization_config=quant,
                                                 device_map="cuda" if use_cuda else None)
    if args.adapter:
        # Without this, PEFT treats a missing local path as a Hub repo id and fails with an auth error.
        if not os.path.isfile(os.path.join(args.adapter, "adapter_config.json")):
            sys.exit(f"no adapter at {args.adapter}: did train_qlora.py finish?")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    items = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)
    weights_gb = torch.cuda.memory_allocated() / 1e9 if use_cuda else 0.0
    rows, gen_tokens, gen_time = [], 0, 0.0
    for i in range(0, len(items), args.batch_size):
        batch = items[i: i + args.batch_size]
        enc = tok([build_prompt(tok, x["prompt"]) for x in batch], return_tensors="pt",
                  padding=True).to(model.device)
        sync()
        t0 = time.perf_counter()  # time generation only, not tokenizing or scoring
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        sync()
        gen_time += time.perf_counter() - t0
        new = out[:, enc["input_ids"].shape[1]:]
        for x, ids in zip(batch, new):
            n = int((ids != tok.pad_token_id).sum())
            gen_tokens += n
            answer = tok.decode(ids, skip_special_tokens=True).strip()
            rows.append({**x, "model_answer": answer, "gen_tokens": n,
                         **score(answer, x["answer"])})
        print(f"{min(i + args.batch_size, len(items))}/{len(items)}", flush=True)

    with open(args.out + ".jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"model": args.model, "adapter": args.adapter, "data": os.path.basename(args.data),
               "quant": args.quant, "batch_size": args.batch_size, **summarize(rows),
               "truncated_pct": round(100 * sum(r["gen_tokens"] >= args.max_new_tokens for r in rows)
                                      / len(rows), 1),
               "gen_tokens_per_s": round(gen_tokens / gen_time, 1),
               "sec_per_batch": round(gen_time / -(-len(rows) // args.batch_size), 2),
               "weights_gb": round(weights_gb, 2),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if use_cuda else 0.0,
               "device": torch.cuda.get_device_name() if use_cuda else "cpu"}
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
