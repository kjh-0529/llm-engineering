# LLM Engineering

Hands-on LLM work, from model to application, built by an HPC and GPU engineer. The companion repository [cuda-hpc-portfolio](https://github.com/kjh-0529/cuda-hpc-portfolio) covers the GPU side: CUDA kernels, PyTorch extensions and multi-GPU training.

| # | Project | What it shows | Status |
|---|---|---|---|
| 1 | [Fine-tuning and inference optimization](01-finetune-inference/) | Data curation, QLoRA fine-tuning of Qwen3-1.7B for Korean technical Q&A, honest evaluation, fast inference on a T4 | In progress |
| 2 | HPC user-support assistant (RAG) | Retrieval over public HPC documentation, evaluation set, step-by-step retrieval improvements, log analysis, web demo | Planned |

## How this was built

This portfolio was developed with [Claude Code](https://claude.com/claude-code), an AI coding assistant, used as a pair programmer. Claude wrote most of the code and first drafts of the documentation; commits show it as author or co-author.

My part:

- **Direction.** I chose what to build and made the calls along the way, for example which projects to keep or replace, which dataset and model size to use, and when a result was good enough to merge.
- **Every measurement.** I ran all GPU benchmarks, training runs and evaluations myself on Kaggle (Tesla T4) and reported the raw output. No number in these READMEs was estimated or filled in; each one comes from a run I executed, and the measured values are committed next to the results as CSV or JSON summaries.
- **Review and domain knowledge.** I read the analyses, questioned results that looked wrong, and bring three years of HPC user support on a national supercomputer. The HPC evaluation set is still a draft awaiting my review.

Before measuring, each step wrote down a prediction; when a result disagreed, the README says so and explains why.
