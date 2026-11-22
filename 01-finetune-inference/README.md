# Fine-tuning and Inference Optimization: Korean Technical Q&A

Fine-tune a small open LLM to answer Korean technical and programming questions, measure the change honestly, then make it fast on a Tesla T4.

Status: data, evaluation and baseline done (re-measured after an audit fixed the data and the Korean metric); QLoRA training script ready; training run next.

## Choices

| | Choice | Why |
|---|---|---|
| Base model | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0) | Text-only and well supported by transformers, PEFT, bitsandbytes and vLLM. The newer Qwen3.5-2B is a multimodal architecture, a risk on Turing GPUs, where some recent model backends have dropped support. Fits a 16 GB T4 for QLoRA. |
| Hardware | Tesla T4 (Kaggle) | No bfloat16 on Turing (sm_75): training and inference use float16. |
| Data | [nlpai-lab/kullm-v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2) (Apache-2.0), Alpaca and Dolly parts only | See below. A Korean code dataset (Evol-Instruct-Code-80k-v1-ko) was rejected because its CC-BY-NC-SA licence would carry over to the fine-tuned model. |

## Data: what was wrong with the raw dataset

KULLM-v2 is a DeepL translation of Alpaca, Dolly and Vicuna. Measured on all 152,630 rows before choosing what to use:

| Source | Rows | Code blocks that kept their line breaks | Other problems |
|---|---|---|---|
| Alpaca | 52,002 | 1,432 / 1,597 (90%) | — |
| Dolly | 15,009 | 12 / 12 | long reference passages in the input |
| Vicuna | 85,619 | **4,286 / 23,047 (19%)** | 6,830+ prompts start mid-conversation ("좋아, 이제 ...") without their context |

Training on Vicuna would teach the model to write code without line breaks and to answer questions it cannot see, so it is left out. [`data/prepare_data.py`](data/prepare_data.py) then:

1. keeps prompts whose **instruction** mentions a technical or programming keyword (matching the whole prompt picked up Dolly passages about TV networks and game studios, and the word "프로그램" matched "TV 프로그램"),
2. drops answers with broken code blocks (32), inputs with code squashed onto one line (27), **code that the translator itself turned into Korean (224)**, very short answers (4,733), very long samples (84) and duplicate prompts (302),
3. splits the 3,116 technical samples into **200 test, 200 validation and 2,716 training**, and adds 1,000 general samples to training so the model keeps its general ability.

The translated-code filter came from an audit after the baseline: DeepL had translated code keywords and names, e.g. `반환 ''.join(...)` for `return`, `int(입력(...))` for `input`, `SELECT * FROM 주택에서 ...`, and code fences labelled `파이썬`. About 3.6% of the earlier training split had such answers; training on them teaches broken code. Detection is a keyword-plus-Hangul heuristic on code lines with strings and comments removed; a random sample of 8 dropped items were all real cases. [`test_fixes.py`](test_fixes.py) checks it.

The processed splits are committed in [`data/`](data/) with [`stats.json`](data/stats.json). Part of the original English data was produced by GPT models; the model card will state this.

**HPC evaluation set.** [`data/hpc_eval_ko.jsonl`](data/hpc_eval_ko.jsonl): 30 Korean questions on Slurm, MPI, OpenMP, CUDA, GPUs, PyTorch DDP, Linux and parallel performance, with reference answers. It checks the domain this portfolio is about, which the general dataset barely covers. Status: draft, to be reviewed by a domain expert (`"reviewed": false`).

## Evaluation

[`eval/evaluate.py`](eval/evaluate.py) generates an answer for every test question (greedy decoding, the same Korean system prompt for every model, Qwen3 thinking mode off) and [`eval/metrics.py`](eval/metrics.py) scores it:

| Metric | What it measures |
|---|---|
| Korean answers (%) | answers whose letters outside code are at least 50% Hangul (vs. Latin); small multilingual models often answer Korean questions in English. Counted only on items whose **reference** is Korean (178 of 200 test items): a correct code-only answer has no letters outside code and would otherwise count as "not Korean" |
| CJK drift (%) | answers containing Chinese characters or Japanese kana outside code |
| chrF | character n-gram overlap with the reference answer; character-level, so it suits Korean spacing and particles |
| Code OK (%) | when the reference has a code block, the answer has one with intact line breaks |
| Tokens/s | generation speed, for the inference part |

No single metric is enough for free-form answers, so results are reported side by side, with sample answers.

A CPU smoke test with Qwen3-0.6B on two HPC questions ran end to end. It answered the Slurm GPU question with `--gres=2` (the correct option is `--gres=gpu:2`) and repeated itself on the second, a sign the HPC set separates good and bad answers.

## Baseline: Qwen3-1.7B as it is

Tesla T4, float16, greedy decoding. Summaries in [`results/`](results/).
Measured on the current split with the fixed metric. Korean metrics count only items whose reference is Korean (178 of 200 test, 28 of 30 HPC).

| Set | Korean answers | Korean ratio | CJK drift | chrF | Code OK | Mean length | Tokens/s |
|---|---|---|---|---|---|---|---|
| Test (200 technical) | 84.8% (of 178) | 0.80 | 0% | 22.2 | 76.9% (of 65) | 302 chars | 90 |
| HPC (30) | 100% (of 28) | 0.85 | 0% | 18.2 | — | 360 chars | 104 |

The numbers look good; the answers do not. On the first five HPC questions:

| Question | Base model answer | Verdict |
|---|---|---|
| Request 2 GPUs in Slurm | `#SBATCH --gres=gpu:2` | correct |
| Why is a job PENDING? | `squeue`, `scontrol show node`, ... then trails off into "`srm.c`" | partly right; misses the REASON column, ends in noise |
| sbatch vs srun | long generic explanation | vague |
| Cancel a running job | `scontrol cancel job <id>` (twice) | **wrong**: the command is `scancel <id>` |
| OUT_OF_MEMORY / oom-kill | `kill -15 <job_id>`, then `scontrol show job` | **wrong**: a job ID is not a process ID; the fix is to check `sacct` MaxRSS and raise `--mem` |

What this means for the plan:

- **Language is mostly fine, with one real gap.** No Chinese or Japanese drift, and every HPC answer is in Korean. On the test set, 15% of answers to questions with a Korean reference are still mostly not Korean. I first suspected a metric bug, since a correct code-only answer used to score 0; the metric was fixed to skip code-only references, the baseline was re-run, and the 15% stayed (85.0% → 84.8%). So the gap is real, and closing it is a concrete goal for fine-tuning.
- **Facts are.** The model confidently invents Slurm commands. The training data is general technical Q&A and contains almost nothing about Slurm or MPI, so **fine-tuning is not expected to fix HPC facts**; it should mainly improve Korean consistency, answer format and fit to the reference style (chrF). Written down before training, so the result can be checked against it.
- That gap is exactly what retrieval is for: project 2 (RAG over HPC documentation) is the fix for wrong facts, and this baseline is the evidence for needing it.
- chrF alone would have hidden all of this: a fluent wrong answer and a fluent right one score about the same. Every comparison below keeps a sample of answers next to the numbers.

## QLoRA fine-tuning

[`train_qlora.py`](train_qlora.py) uses plain transformers + PEFT + bitsandbytes (no extra training framework).

| Setting | Value | Why |
|---|---|---|
| Base weights | 4-bit NF4, double quantization, frozen | fits the 16 GB T4 with room for activations |
| Compute | float16, gradient checkpointing | the T4 has no bfloat16 |
| LoRA | r 16, alpha 32, dropout 0.05, on q/k/v/o and gate/up/down projections | standard QLoRA setup; about 1.7% of parameters trainable |
| Batch | 4 per step x 4 accumulation = 16 sequences, max 1,024 tokens | |
| Schedule | 2 epochs, learning rate 2e-4, cosine, 3% warm-up, paged 8-bit AdamW | |
| Loss | answer tokens only | the model learns to answer, not to repeat prompts |

Prompts are built with the evaluation script's own `build_prompt` (same system prompt, Qwen3 thinking mode off), so training and evaluation see exactly the same format. Prompt and answer are tokenized separately, so the loss mask boundary is exact, and the script asserts on the first example that the unmasked labels decode to its answer.

Checked before the GPU run: a CPU smoke test with Qwen3-0.6B (2 steps, 8 examples) trained, evaluated, saved the adapter, and `eval/evaluate.py --adapter` loaded it. Two arguments used in older tutorials, `warmup_ratio` and `group_by_length`, no longer exist in transformers 5.x; the script computes warm-up steps itself so it runs on both 4.x and 5.x.

## Plan

1. ~~Data and evaluation~~
2. ~~Baseline: score Qwen3-1.7B as it is~~
3. QLoRA fine-tuning (4-bit base, float16 compute on the T4)
4. Before/after comparison, with failure analysis
5. Inference optimization: float16 vs 8-bit vs 4-bit, Hugging Face generate vs vLLM; latency, throughput, memory
6. Model card

## Run

```bash
pip install -r requirements.txt
python data/prepare_data.py            # optional: the splits are already committed
python eval/evaluate.py --model Qwen/Qwen3-1.7B --data data/test.jsonl --out results/base_test
python eval/evaluate.py --model Qwen/Qwen3-1.7B --data data/hpc_eval_ko.jsonl --out results/base_hpc
python train_qlora.py                  # writes the adapter to runs/qlora
python eval/evaluate.py --adapter runs/qlora --data data/test.jsonl --out results/qlora_test
python eval/evaluate.py --adapter runs/qlora --data data/hpc_eval_ko.jsonl --out results/qlora_hpc
```
