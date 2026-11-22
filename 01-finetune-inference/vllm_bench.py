"""Serve the merged model with vLLM and measure it the same way as eval/evaluate.py.

    pip install vllm
    python vllm_bench.py --out results/vllm_all                       # all 200 at once
    python vllm_bench.py --max-num-seqs 16 --out results/vllm_b16     # same concurrency as HF batch 16
    python vllm_bench.py --sequential --limit 20 --out results/vllm_lat        # one request at a time
    python vllm_bench.py --sequential --limit 20 --eager --out results/vllm_lat_eager  # no CUDA graphs

Same prompts (eval's build_prompt), greedy decoding, 512 new tokens, same metrics.
vLLM batches requests continuously and reserves GPU memory up front, so peak
memory is not comparable to the Hugging Face runs and is not reported.
"""
import argparse
import json
import os
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))
from evaluate import build_prompt  # noqa: E402
from metrics import score, summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(HERE, "runs", "merged"))
ap.add_argument("--data", default=os.path.join(HERE, "data", "test.jsonl"))
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--max-num-seqs", type=int, default=256, help="max requests batched together")
ap.add_argument("--sequential", action="store_true", help="send one request at a time (latency)")
ap.add_argument("--eager", action="store_true", help="disable CUDA graphs")
ap.add_argument("--max-new-tokens", type=int, default=512)
args = ap.parse_args()

items = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.limit]
tok = AutoTokenizer.from_pretrained(args.model)
prompts = [build_prompt(tok, x["prompt"]) for x in items]

# T4: no bfloat16, so float16; prompts are at most ~1k tokens plus 512 new ones.
llm = LLM(args.model, dtype="half", max_model_len=2048, gpu_memory_utilization=0.85,
          max_num_seqs=args.max_num_seqs, enforce_eager=args.eager, seed=0)
params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
llm.generate(prompts[:1], params, use_tqdm=False)  # warm-up, excluded from timing

t0 = time.perf_counter()
if args.sequential:
    outs = [llm.generate([p], params, use_tqdm=False)[0] for p in prompts]
else:
    outs = llm.generate(prompts, params)
elapsed = time.perf_counter() - t0

rows = []
for x, o in zip(items, outs):
    answer, n = o.outputs[0].text.strip(), len(o.outputs[0].token_ids)
    rows.append({**x, "model_answer": answer, "gen_tokens": n, **score(answer, x["answer"])})
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out + ".jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
summary = {"engine": "vllm", "model": args.model, "data": os.path.basename(args.data),
           "max_num_seqs": args.max_num_seqs, "sequential": args.sequential, "eager": args.eager,
           **summarize(rows),
           "truncated_pct": round(100 * sum(r["gen_tokens"] >= args.max_new_tokens for r in rows)
                                  / len(rows), 1),
           "gen_tokens_per_s": round(sum(r["gen_tokens"] for r in rows) / elapsed, 1),
           "sec_per_answer": round(elapsed / len(rows), 2), "total_seconds": round(elapsed, 1)}
with open(args.out + ".summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
