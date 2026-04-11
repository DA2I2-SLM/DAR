# Diversity-Aware Retention (DAR)
Source code for the paper: **"Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention"**
- Preprint: [arXiv:2603.20640](https://arxiv.org/abs/2603.20640)

## Introduction / Abstract
> **[TODO: We will update this section to explain the core mechanism, motivation, and main contributions of FilterMAD.]**

## Requirements Setup (using Conda)

```bash
conda create -n filter_mad python=3.10.16 -y
conda activate filter_mad
pip install -r requirements.txt
```

> **Note on Environment Variables:**
> If you are using gated models (like Llama-3.1 or Gemma), make sure you have requested access on Hugging Face and exported your token before running any scripts:
> ```bash
> export HF_TOKEN="your_huggingface_token_here"
> ```

## Project Structure
```text
.
├── data/              # Stores raw benchmark datasets (downloaded and processed)
├── out/               # TSV metric summaries and JSONL logs of complete debate history
├── result/            # Runtime operation and token logs
├── scripts/           # Collection of bash scripts for running large-scale experiments
└── src/
    ├── main.py        # Main entry point pipeline for Multi-Agent Debate
    ├── dev.py         # Filtering algorithms (filter_critical, filter_support, etc.)
    ├── evaluator.py   # Code for metric evaluation and regex parsing of outcomes
    └── model/         # vLLM initializations and sampling configurations
```

## Data Preparation
Datasets are handled through `data/data_utils.py`. Standard benchmarks are automatically loaded when you pass their name to the `--data` argument in the execution command. 

## Output & Logs
When an experiment finishes:
- **Accuracy Metrics**: Total accuracies across rounds are appended to `out/<dataset>_vllm_batch_logs.tsv`.
- **Debate History**: Detailed traces including agent peers, generated text, uncertainty scores (ANLL), and final answers are serialized to `out/history/<experiment_name>.jsonl`. For debugging mode, `--debug` prepends "DEBUG_" to the filename.

## Experiments & How to Run

The explicit run commands for each full dataset benchmark are provided in `scripts/`. Below are basic usage examples.

For example, to run the Arithmetics dataset on Qwen2.5-3b-Instruct:
```bash
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2
```
To run Sparse or Centralized topologies, append `--sparse` or `--centralized`. To enable heterogeneous agent personas, append `--multi_persona`.

#### 1. List of models

- `Qwen2.5-1.5B, 3B, 7B`
- `Llama3.1-8B`
- `Falcon3-7B`

#### 2. Supported benchmarks

- Math: Arithmetics, GSM8K
- QA: MMLU (Form. Log.), HH-RLHF
- Others: HellaSwag, MMLU (Pro. Medicine), CommonSenseQA

#### 3. Baselines

```bash
# Base
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2

# Filter top 50% most certain answers
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --top_k_uncertainty 0.5

# Uncertain Prompt
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True

# Vote Prompt
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --vote_prompt True

# Our Method: Vote Prompt + Filter Critical
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical

# Ours Method: Huggingface Model (no vLLM) (adjust batch size with `--hf_batch_size` for memory constraints)
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical --use_hf_inference --hf_batch_size 16
```

#### 4. Analysis
```bash
# Retaining criteria
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_certain
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_support
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_nonvote

# Separate Moderator
python src/main.py --model qwen2.5-3b --num_agents 4 --data arithmetics --data_size 100 --debate_rounds 2 --uncertainty_prompt True --vote_prompt True --m_role filter_critical --separate_moderator qwen2.5-1.5b
```

#### 5. Simple Run for validation

This script will run a quick validation of the entire method with Qwen2.5-3B with one fixed seed. Note that due to non-determinism in sampling (e.g., vLLM back-end), results may vary slightly across runs. Thats why we run multiple seeds and report averages with standard deviations in the paper.
```bash
python scripts/validate.sh
```

## Upcoming Features
* Support new benchmarks, e.g. AIME24, AIME25.

## Citations
```bash
@article{nguyen2026hear,
  title={Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention},
  author={Nguyen, Manh and Nguyen, Anh and Nguyen, Dung and Venkatesh, Svetha and Le, Hung},
  journal={arXiv preprint arXiv:2603.20640},
  year={2026}
}
```

## Acknowledgements

* [1] [Debate or Vote](https://github.com/deeplearning-wisc/debate-or-vote)
* [2] [vLLM](https://github.com/vllm-project/vllm)