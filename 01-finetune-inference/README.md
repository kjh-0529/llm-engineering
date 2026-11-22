# Fine-tuning and Inference Optimization: Korean Technical Q&A

Fine-tune a small open LLM to answer Korean technical and programming questions, measure the change honestly, then make it fast on a Tesla T4.

Status: step 1 (data and evaluation) done; baseline measurement next.

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
2. drops answers with broken code blocks (32), inputs with code squashed onto one line (27), very short answers (4,733), very long samples (84) and duplicate prompts (302),
3. splits the 3,305 technical samples into **200 test, 200 validation and 2,905 training**, and adds 1,000 general samples to training so the model keeps its general ability.

The processed splits are committed in [`data/`](data/) with [`stats.json`](data/stats.json). Part of the original English data was produced by GPT models; the model card will state this.

**HPC evaluation set.** [`data/hpc_eval_ko.jsonl`](data/hpc_eval_ko.jsonl): 30 Korean questions on Slurm, MPI, OpenMP, CUDA, GPUs, PyTorch DDP, Linux and parallel performance, with reference answers. It checks the domain this portfolio is about, which the general dataset barely covers. Status: draft, to be reviewed by a domain expert (`"reviewed": false`).

## Evaluation

[`eval/evaluate.py`](eval/evaluate.py) generates an answer for every test question (greedy decoding, the same Korean system prompt for every model, Qwen3 thinking mode off) and [`eval/metrics.py`](eval/metrics.py) scores it:

| Metric | What it measures |
|---|---|
| Korean answers (%) | answers whose letters outside code are at least 50% Hangul (vs. Latin); small multilingual models often answer Korean questions in English |
| CJK drift (%) | answers containing Chinese characters or Japanese kana outside code |
| chrF | character n-gram overlap with the reference answer; character-level, so it suits Korean spacing and particles |
| Code OK (%) | when the reference has a code block, the answer has one with intact line breaks |
| Tokens/s | generation speed, for the inference part |

No single metric is enough for free-form answers, so results are reported side by side, with sample answers.

A CPU smoke test with Qwen3-0.6B on two HPC questions ran end to end. It answered the Slurm GPU question with `--gres=2` (the correct option is `--gres=gpu:2`) and repeated itself on the second, a sign the HPC set separates good and bad answers.

## Plan

1. ~~Data and evaluation~~
2. Baseline: score Qwen3-1.7B as it is
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
```
