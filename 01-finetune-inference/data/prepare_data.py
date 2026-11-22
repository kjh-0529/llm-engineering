"""Build the Korean technical Q&A dataset from KULLM-v2.

Source: nlpai-lab/kullm-v2 (Apache-2.0), a DeepL translation of Alpaca, Dolly
and Vicuna data. Measured on the raw file (152,630 rows):

  source   rows    code blocks that kept their newlines
  alpaca   52,002  1,432 / 1,597  (90%)
  dolly    15,009  12 / 12
  vicuna   85,619  4,286 / 23,047 (19%), plus 6,830+ prompts that start
                   mid-conversation ("좋아, 이제 ...") without their context

So this script keeps Alpaca and Dolly only, then:
  1. keeps technical / programming prompts (keyword filter on the instruction)
  2. optionally adds a random sample of general prompts (--general)
  3. drops samples whose code lost its newlines (in answer code blocks or in a
     code input), empty or very short answers, very long samples, and
     duplicate prompts
  4. splits into train / val / test with a fixed seed

Output: train.jsonl, val.jsonl, test.jsonl with {"id", "prompt", "answer",
"category"} and stats.json, in --out-dir.
"""
import argparse
import collections
import json
import os
import random
import re

from huggingface_hub import hf_hub_download

TECH = re.compile(
    r"(코드|함수|파이썬|python|자바|java\b|c\+\+|자바스크립트|javascript|프로그래밍|알고리즘|"
    r"리눅스|linux|유닉스|unix|sql|쿼리|gpu|cpu|컴파일|스크립트|클래스|배열|정규식|서버|"
    r"데이터베이스|api\b|병렬|스레드|메모리|운영체제|네트워크|프로토콜|html|css|git\b|"
    r"머신러닝|딥러닝|신경망|```)",
    re.IGNORECASE,
)
CODE_BLOCK = re.compile(r"```(.*?)```", re.DOTALL)


def code_blocks_ok(text):
    """False if any fenced code block lost its line breaks in translation."""
    return all("\n" in block.strip("\n ") or len(block) < 40 for block in CODE_BLOCK.findall(text))


CODE_CHARS = re.compile(r"[;{}]|==|\bdef |\bint |\breturn\b")


def input_code_broken(text):
    """True if the input looks like source code squashed onto one line."""
    return "\n" not in text and len(text) > 60 and len(CODE_CHARS.findall(text)) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--general", type=int, default=1000,
                    help="general (non-technical) samples to add to train, to keep general ability")
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = hf_hub_download("nlpai-lab/kullm-v2", "kullm-v2.jsonl", repo_type="dataset")
    raw = [json.loads(line) for line in open(path, encoding="utf-8")]
    stats = collections.OrderedDict(raw_rows=len(raw))

    rows = [r for r in raw if r["id"].split("_")[0] in ("alpaca", "dolly")]
    stats["alpaca_dolly_rows"] = len(rows)

    drops = collections.Counter()
    seen = set()
    tech, general = [], []
    for r in rows:
        prompt = r["instruction"].strip()
        if r["input"].strip():
            prompt += "\n\n" + r["input"].strip()
        answer = r["output"].strip()
        if len(answer) < 20:
            drops["answer_too_short"] += 1
            continue
        if len(prompt) + len(answer) > args.max_chars:
            drops["too_long"] += 1
            continue
        if not code_blocks_ok(answer):
            drops["broken_code_block"] += 1
            continue
        if input_code_broken(r["input"].strip()):
            drops["broken_code_in_input"] += 1
            continue
        if prompt in seen:
            drops["duplicate_prompt"] += 1
            continue
        seen.add(prompt)
        item = {"id": f"{r['id'].split('_')[0]}-{len(seen)}", "prompt": prompt, "answer": answer}
        # Match the instruction only: Dolly inputs are long reference passages
        # whose wording would otherwise trigger the keywords.
        (tech if TECH.search(r["instruction"]) else general).append(item)
    stats["dropped"] = dict(drops)
    stats["technical_candidates"] = len(tech)

    rng = random.Random(args.seed)
    rng.shuffle(tech)
    test = tech[: args.test_size]
    val = tech[args.test_size: args.test_size + args.val_size]
    train = tech[args.test_size + args.val_size:]
    for x in test + val + train:
        x["category"] = "technical"
    extra = rng.sample(general, min(args.general, len(general)))
    for x in extra:
        x["category"] = "general"
    train = train + extra
    rng.shuffle(train)
    stats.update(train=len(train), train_technical=len(train) - len(extra),
                 train_general=len(extra), val=len(val), test=len(test))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for x in split:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
