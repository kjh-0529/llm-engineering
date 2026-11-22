# Fine-tuning and Inference Optimization: Korean Technical Q&A

Fine-tune a small open LLM to answer Korean technical and programming questions, measure the change honestly, then make it fast on a Tesla T4.

Status: data, evaluation and baseline done (re-measured after an audit fixed the data and the Korean metric); QLoRA fine-tuning done, evaluated and [published](https://huggingface.co/kjh0529/qwen3-1.7b-ko-tech-qlora); inference optimization measured on the T4 (vLLM next).

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
| Batch | 2 per step x 8 accumulation = 16 sequences, max 1,024 tokens | batch 4 ran out of memory on the T4: with a 152k-token vocabulary, the logits for 4 x 1,024 tokens need about 4 GiB more during the loss |
| Schedule | 2 epochs, learning rate 2e-4, cosine, 3% warm-up, paged 8-bit AdamW | |
| Loss | answer tokens only | the model learns to answer, not to repeat prompts |

Prompts are built with the evaluation script's own `build_prompt` (same system prompt, Qwen3 thinking mode off), so training and evaluation see exactly the same format. Prompt and answer are tokenized separately, so the loss mask boundary is exact, and the script asserts on the first example that the unmasked labels decode to its answer.

Checked before the GPU run: a CPU smoke test with Qwen3-0.6B (2 steps, 8 examples) trained, evaluated, saved the adapter, and `eval/evaluate.py --adapter` loaded it. Two arguments used in older tutorials, `warmup_ratio` and `group_by_length`, no longer exist in transformers 5.x; the script computes warm-up steps itself so it runs on both 4.x and 5.x.

## Results after fine-tuning

Training on the T4: 234 updates (2 epochs) in 1 h 26 min. Validation loss fell from 2.057 (step 100) to 2.003 (step 200) and 2.001 (end) and never rose, so there was no overfitting; a third epoch would add little. Loss curve: [`results/qlora_training_loss.csv`](results/qlora_training_loss.csv).

**Technical test set (200 questions)**

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Korean answers (of 178 with a Korean reference) | 84.8% | **93.3%** | +8.5 points |
| chrF | 22.2 | **27.9** | +26% |
| Code OK (of 65) | 76.9% | **81.5%** | +4.6 points |
| Mean answer length | 302 chars | 404 chars | +34% |
| Tokens/s (unmerged adapter) | 90 | 79 | −12% |

**HPC set (30 questions)**

| Metric | Base | Fine-tuned |
|---|---|---|
| Korean answers | 100% | 100% |
| chrF | 18.2 | **12.9** |
| Mean answer length | 360 chars | **614 chars** |

**Same five HPC questions as the baseline**

| Question | Base | Fine-tuned |
|---|---|---|
| Request 2 GPUs | `--gres=gpu:2`, correct | correct, plus a redundant explanation |
| Why is a job PENDING? | `squeue`, `scontrol`, then noise | invents "reservation" problems; no longer mentions `squeue` |
| sbatch vs srun | vague | vague, and repeats the same sentence twice |
| Cancel a job | `scontrol cancel job` (wrong) | `scontrol cancel` (still wrong), now with invented details ("waits ... default 10 seconds") |
| OUT_OF_MEMORY | `kill <job_id>` (wrong) | generic steps, no concrete command |

What it shows, checked against the prediction written before training:

- **Language and format improved, as predicted.** More Korean answers, closer to the reference style, more intact code blocks on the in-distribution test set.
- **HPC facts did not improve, as predicted**, and the answers got worse in a way I had not predicted: longer and more elaborate when wrong. The training answers are long, explanatory Alpaca-style texts, so the model learned to write more; where it has no knowledge, it now fills the extra length with plausible invented detail. chrF on the HPC set falls accordingly.
- **Lesson:** fine-tuning on general data changes how a model answers, not what it knows. Wrong facts need a source of facts, which is what retrieval (project 2) adds.
- The speed drop comes from the unmerged LoRA adapter and longer answers; the inference step addresses it.

Not measured yet: how many answers hit the 512-token limit (longer answers make this more likely; the audit flagged that it is not tracked), and whether a repetition penalty or shorter training answers would curb the verbosity.

## Inference optimization

Same T4, same 200 test questions, greedy decoding, so speed, memory and quality are measured together. Each run is one `eval/evaluate.py` call:

| Run | How | Question |
|---|---|---|
| Adapter, unmerged | `--adapter runs/qlora` | the starting point |
| Merged, float16 | [`merge_adapter.py`](merge_adapter.py), then `--model runs/merged` | what does the LoRA side path cost? |
| Merged, 8-bit | `--model runs/merged --quant 8bit` (bitsandbytes LLM.int8) | memory vs. speed vs. quality |
| Merged, 4-bit | `--model runs/merged --quant 4bit` (NF4) | the same, one step further |

Each runs twice: batch 16 over all 200 questions (throughput and quality) and batch 1 over 20 questions (latency of a single request).

What `evaluate.py` now records, beyond the quality metrics: tokens/s timed around `generate()` only (earlier numbers also included tokenizing and scoring), seconds per batch, GPU memory for the weights and at peak, and the share of answers cut off at the 512-token limit. Merging was checked on CPU: the merged model gives exactly the same greedy answers as base plus adapter.

Predictions, written before measuring:

- **Merging** removes the extra LoRA matrix multiplies in every layer, so it should recover most of the 12% speed lost to the adapter, with identical quality.
- **8-bit** should roughly halve weight memory (about 3.4 → 1.9 GB) but run slower: LLM.int8 handles outlier features in float16, which costs time.
- **4-bit** should cut weight memory to about a third. At batch 1, where decoding is limited by reading weights from memory, it may come close to float16; at batch 16 the dequantization overhead should make it slower. A small quality drop is possible.
- vLLM (continuous batching, paged KV cache) comes next, in a separate run, since installing it changes PyTorch.

### Results

Tesla T4, float16 compute. Raw values: [`results/inference_t4.csv`](results/inference_t4.csv).

**Throughput: batch 16, all 200 test questions**

| Run | Tokens/s | Weights | Peak memory | Korean answers | chrF | Code OK | Cut off at 512 |
|---|---|---|---|---|---|---|---|
| Adapter, unmerged | 79.1 | 3.51 GB | 4.80 GB | 93.3% | 27.9 | 81.5% | 13.5% |
| **Merged, float16** | **123.8** | 3.44 GB | 4.73 GB | 93.8% | 28.0 | 84.6% | 14.0% |
| Merged, 8-bit | 33.5 | 2.03 GB | 3.33 GB | 94.4% | 28.0 | 83.1% | 15.5% |
| **Merged, 4-bit** | 106.4 | **1.42 GB** | **2.71 GB** | 98.3% | 28.8 | 84.6% | 16.0% |

**Latency: batch 1, 20 questions**

| Run | Tokens/s | Seconds per answer | Peak memory |
|---|---|---|---|
| Adapter, unmerged | 11.9 | 19.8 | 4.37 GB |
| **Merged, float16** | **21.4** | **11.7** | 3.56 GB |
| Merged, 8-bit | 5.9 | 34.5 | 2.16 GB |
| Merged, 4-bit | 15.9 | 14.6 | 1.53 GB |

Checked against the predictions:

| Prediction | Result | |
|---|---|---|
| Merging recovers the ~12% lost to the adapter | **+57% at batch 16, +80% at batch 1** | far more than predicted |
| 8-bit halves weight memory and runs slower | −41% memory, **3.7x slower** | right direction, slower than expected |
| 4-bit cuts weights to about a third; slower at batch 16 | −59% memory; 86% of float16 speed at batch 16 | right |
| 4-bit comes close to float16 at batch 1 | **74%** of float16 speed | wrong |

What it shows:

- **The unmerged adapter was the biggest cost.** It adds two small matrix multiplies, a scaling and an addition to each of the 7 projections in every layer: hundreds of extra small GPU kernels per generated token. Merging folds them into the base weights at no quality cost (identical answers on the CPU check; chrF 27.9 vs. 28.0 here, from float16 rounding of the merged weights). The earlier "12% slower than the base model" compared timings taken a different way and understated the cost.
- **Batch-1 decoding here is limited by overhead, not memory bandwidth.** At 21 tokens/s, float16 reads its 3.4 GB of weights about 21 times a second, roughly 75 GB/s, a quarter of the T4's 320 GB/s. So reading 4-bit weights saves little, while unpacking them costs time: 4-bit is slower, not close. The bottleneck is the thousands of small kernel launches and Python overhead per token in `generate()`. CUDA graphs and a dedicated serving engine (vLLM, next) target exactly this.
- **Batching is the cheapest speed-up.** Float16 throughput rises 5.8x from batch 1 to batch 16 (21 → 124 tokens/s), at the cost of longer waits per request.
- **8-bit is not worth it on a T4.** LLM.int8 handles outlier features separately in float16; on this card it is 3.7x slower than float16 for 41% less memory, while 4-bit saves more memory and is 3x faster than 8-bit.
- **No quality loss from quantization was detected.** 4-bit scored as high as or higher than float16 (Korean answers 98.3% vs. 93.8%). I do not have an explanation for the higher Korean rate and would not claim 4-bit is better: quantization changes greedy decoding paths, and the answers have not been compared by hand.
- **14–16% of answers hit the 512-token limit**, in every configuration. This affects chrF and the other quality metrics alike, so it does not change the comparison, but the absolute quality numbers are measured on partly truncated answers.

**Recommendation for this model on a T4:** serve the merged float16 model when memory allows (it uses under 5 GB of the 16 GB); use 4-bit when memory is tight, for 14–26% less speed and no measured quality loss; avoid 8-bit.

### vLLM

[`vllm_bench.py`](vllm_bench.py) serves the merged float16 model with vLLM, using the same prompts, greedy decoding, token limit and metrics as `evaluate.py`. vLLM schedules requests continuously (a finished answer frees its slot for the next request at once), keeps the attention cache in fixed-size pages, and by default replays each decoding step as a captured CUDA graph instead of launching its kernels one by one.

| Run | Compared with |
|---|---|
| All 200 requests at once | Hugging Face batch 16 (123.8 tokens/s) |
| At most 16 at a time (`--max-num-seqs 16`) | the same concurrency as Hugging Face batch 16 |
| One request at a time, 20 questions | Hugging Face batch 1 (21.4 tokens/s) |
| One at a time without CUDA graphs (`--eager`) | isolates what CUDA graphs add |

Predictions, written before measuring:

- **One at a time** should be the largest gain. The Hugging Face batch-1 run used only about a quarter of the T4's memory bandwidth because of per-token overhead; CUDA graphs remove most of it. The ceiling is reading 3.4 GB of weights per token at 320 GB/s, about 90 tokens/s, so 3x or more over 21.4 tokens/s is plausible. Without CUDA graphs much of the gain should disappear.
- **At 16 at a time**, continuous batching should beat static batches, where finished answers sit idle until the slowest one in the batch ends: a clear gain, smaller than for single requests.
- **All at once** should give the highest throughput, several times Hugging Face batch 16.
- Quality should match the Hugging Face float16 run up to small differences from different kernels.

## Plan

1. ~~Data and evaluation~~
2. ~~Baseline: score Qwen3-1.7B as it is~~
3. ~~QLoRA fine-tuning (4-bit base, float16 compute on the T4)~~
4. ~~Before/after comparison, with failure analysis~~
5. Inference optimization: ~~float16 vs 8-bit vs 4-bit with Hugging Face generate~~; vLLM next
6. Model card: [`MODEL_CARD.md`](MODEL_CARD.md), uploaded with [`upload_adapter.py`](upload_adapter.py)

## Run

```bash
pip install -r requirements.txt
python data/prepare_data.py            # optional: the splits are already committed
python eval/evaluate.py --model Qwen/Qwen3-1.7B --data data/test.jsonl --out results/base_test
python eval/evaluate.py --model Qwen/Qwen3-1.7B --data data/hpc_eval_ko.jsonl --out results/base_hpc
pip install bitsandbytes                # not preinstalled on Kaggle
python train_qlora.py                   # writes the adapter to runs/qlora
pip uninstall -y torchao                # Kaggle's torchao 0.10 makes recent PEFT fail when loading the adapter
python eval/evaluate.py --adapter runs/qlora --data data/test.jsonl --out results/qlora_test
python eval/evaluate.py --adapter runs/qlora --data data/hpc_eval_ko.jsonl --out results/qlora_hpc
python upload_adapter.py                # needs HF_TOKEN; uploads the adapter with MODEL_CARD.md
python merge_adapter.py                 # writes runs/merged
python eval/evaluate.py --model runs/merged --quant 4bit --data data/test.jsonl --out results/inf_merged_4bit
```
