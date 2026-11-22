"""Merge the LoRA adapter into the base weights and save a standalone model.

    python merge_adapter.py          # writes runs/merged (float16, ~3.4 GB)

A merged model runs without PEFT's extra adapter matmuls, and it can be
quantized (eval/evaluate.py --quant 8bit|4bit) or served by vLLM like any
Hugging Face model. Merging happens in float16 on the GPU (float32 on CPU).
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--adapter", default=os.path.join(HERE, "runs", "qlora"))
ap.add_argument("--out", default=os.path.join(HERE, "runs", "merged"))
args = ap.parse_args()

if not os.path.isfile(os.path.join(args.adapter, "adapter_config.json")):
    raise SystemExit(f"no adapter at {args.adapter}: did train_qlora.py finish?")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
model.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
print(f"merged model saved to {args.out}")
