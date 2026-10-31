# Temporal Evidence-Aware Multi-Agent Reasoning for E-Commerce Dispute Verdict

English | [简体中文](README.zh-CN.md) ![README views](https://komarev.com/ghpvc/?username=Chronojury&label=README+views&color=blue&style=flat)

ChronoJury processes chronological dispute evidence through two modules:

- **SWER — Stage-Wise Evidence Reasoning:** organize evidence into `T` stages, retrieve and inspect evidence, and generate juror verdicts at each stage. Jurors exchange opinions through a directed following network.
- **STVA — Spatio-Temporal Verdict Aggregation:** encode juror verdicts and stage-specific case summaries, aggregate graph information, and model opinion evolution with causal temporal attention to produce a final verdict.

## 1. Repository layout

```text
chronojury/
  swer/                   # Evidence reasoning and juror interactions
  stva/                   # Feature encoding and verdict aggregation
  explanation/            # GNNExplainer reports and case analysis
  llm.py                  # Model endpoint routing
  runtime.py              # Concurrency management
  tracing.py              # Intermediate reasoning records
run.sh                    # Pipeline entry point
requirements.txt          # Environment dependencies
pyproject.toml            # Package installation
```

## 2. Installation

The environment uses Python 3.10, PyTorch 2.7.1 and CUDA 12.8. The server needs an NVIDIA driver that supports CUDA 12.8. Run the following from the repository root:

```bash
conda create -n chronojury python=3.10 pip -y
conda activate chronojury
python -m pip install -r requirements.txt -e .
```

Conda creates the environment. Pip installs the dependencies and project in one command.

## 3. Dataset: VerdictBench

ChronoJury uses [VerdictBench](https://huggingface.co/datasets/piggi/VerdictBench) on Hugging Face, introduced in [CyberJurors (Sun et al., 2026)](https://arxiv.org/abs/2605.28369). It contains 6,000 real e-commerce disputes across five product categories, with transaction information, buyer/seller claims, and multi-round text, image and video evidence. The dataset uses category- and vote-margin-based stratification with an approximate 3:1:2 train/validation/test ratio.

[VerdictBench](https://huggingface.co/datasets/piggi/VerdictBench) requires an access request and specifies **CC BY-NC 4.0** for academic research.

### 3.1 Labels and reference verdicts

Training and evaluation use `case_data.whowins` directly, without relabeling:

| Value | Meaning |
|---|---|
| `0` | The platform verdict supports the buyer. |
| `1` | The platform verdict supports the seller. |

VerdictBench (ICML 2026) is constructed from real-world disputes. Each case is reviewed by human jurors, and the disputant receiving more votes is declared the winner. If the buyer wins, a refund is issued. Otherwise, the transaction is completed. If either disputant challenges the verdict, the platform arranges a second review. In this setting, voting margins can indicate dispute difficulty level. For example, a 9:8 vote suggests that both disputants have valid claims supported by jurors, making the case more challenging to adjudicate. All verdicts are collected only after the final decisions have been reached, ensuring that they reflect fair and correct outcomes.

```bibtex
@article{sun2026cyberjurors,
  title={CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict},
  author={Sun, Yanhui and Liu, Wu and Ming, Haifeng and Wang, Xinru and Yao, Hantao and Zhang, Yongdong},
  journal={arXiv preprint arXiv:2605.28369},
  year={2026}
}
```

### 3.2 Local file organization

Set the local root directory:

```bash
export CHRONOJURY_LOCAL_ROOT="/path/to/chronojury-local"
```

```text
$CHRONOJURY_LOCAL_ROOT/workspace/
  inputs/
    train/<case_id>/data.json
    val/<case_id>/data.json
    test/<case_id>/data.json
  dataset/<case_id>/...
  models/<encoder>/...
```

Each `data.json` retains VerdictBench’s `id`, case metadata, `buyer_claim`, `seller_claim`, `case_details.evidences` and `whowins`. Evidence entries retain `identity` (`买家` or `卖家`) and `content`, including fields such as `text`, `image_paths` and `video_paths`.

### 3.3 Scope of binary verdicts

VerdictBench does not include cases adjudicated with partial refunds or shared responsibility. This reflects the dispute-resolution mechanism of the source platform, whose final verdicts determine only whether the buyer or seller wins. When the parties reach a settlement themselves, such as a partial refund or shared responsibility, the dispute is typically withdrawn without a final verdict and therefore falls outside the scope of this study.

## 4. Model endpoints

All language-model calls use an OpenAI-compatible chat-completions endpoint. Choose one option below and set `CHRONOJURY_MODEL` to the multimodal model served by your endpoint.

### 4.1 Hosted API

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_BASE_URL='https://api.example.com/v1'
export CHRONOJURY_MODEL='your-vision-model'
```

### 4.2 Self-hosted models

Requests can run concurrently across model replicas on multiple GPUs. Replace the IP and port placeholders with your service addresses. All replicas must serve the same model.

```bash
export OPENAI_API_KEY='EMPTY'
export CHRONOJURY_API_BASES='http://<SERVER_IP_1>:<PORT_1>/v1,http://<SERVER_IP_2>:<PORT_2>/v1'
export CHRONOJURY_MODEL='your_model_name'
```

Use `EMPTY` only for services without API-key authentication. `CHRONOJURY_API_BASES` takes precedence over `OPENAI_BASE_URL`. Run `unset CHRONOJURY_API_BASES` to return to a single endpoint. The client supports concurrent requests, CPU data prefetching and temporary avoidance of failing replicas.

## 5. Workflow

Activate the environment, prepare the data and local encoder weights, and configure a model endpoint. From the repository root, run:

```bash
bash run.sh reason       # Generate stage-wise juror verdicts
bash run.sh summarize    # Generate stage-specific case summaries
bash run.sh encode       # Encode juror and case nodes
bash run.sh train        # Train STVA
bash run.sh test         # Evaluate the saved checkpoint
bash run.sh help         # Show usage information
```

Alternatively, `bash run.sh all` runs these five steps in order. Outputs are saved under `$CHRONOJURY_LOCAL_ROOT/results/`. Reasoning, summarization and encoding process the train, validation and test splits. Reasoning and summarization skip existing results by default.

Each stage's case summary uses the evidence available up to that stage, together with case metadata and the buyer/seller claims. Later-stage evidence is excluded. Jurors receive their own previous opinions and the original opinions of their followed neighbors.

The default encoder is `chinese-roberta-wwm-ext-large`, producing 1024-dimensional features. Qwen encoding is also supported. Select it with `CHRONOJURY_ENCODER_BACKEND=qwen`, set `CHRONOJURY_ENCODER_PATH` to its local weights, and match `CHRONOJURY_INPUT_DIM` to the encoder's output width. The reasoning model and text encoder are configured independently.

Keep the encoder, graph and model settings consistent across training, evaluation and explanation. Configuration is maintained in [SWER config](chronojury/swer/config.py) and [STVA config](chronojury/stva/config.py).

## 6. GNNExplainer reports

Reports explain verdicts through **information aggregation within each stage** and **information transfer across stages**, using the trained model without retraining.

```bash
bash run.sh explain --case-id 1            # Generate spatial and temporal explanations
bash run.sh explain --case-id 1 --summary  # Also generate an auxiliary text summary
```

Reports trace the model's actual computation:

1. **Load inputs:** restore a trained model checkpoint, the case's stage-wise node features and its following network, then compute the model's original verdict.
2. **Learn message importance:** freeze model parameters and learn masks for spatial edges and cross-stage temporal messages together, targeting the model's own prediction without ground-truth labels.
3. **Locate key connections:** extract spatial critical paths within each stage and important temporal connections from earlier to later stages of the same node.
4. **Generate the report:** combine key connections with the relevant jurors' original opinions and case summaries to explain the verdict and information flow across stages.

The structured report requires no additional LLM call. Add `--summary` for an auxiliary text summary, which does not participate in the verdict. Path importance explains the model's prediction and does not establish causality.

## 7. Case studies

The three case studies include dispute evidence, juror social relationships, each juror's verdict and rationale across three rounds, and spatial and temporal GNNExplainer explanations.

| Case | Model verdict | Report |
|---|---|---|
| 702 · Watch description dispute | Buyer — correct | [English](docs/case-studies/case_702/README.md) · [中文](docs/case-studies/case_702/README.zh-CN.md) |
| 2588 · Damage disclosure for a secondhand book | Seller — correct | [English](docs/case-studies/case_2588/README.md) · [中文](docs/case-studies/case_2588/README.zh-CN.md) |
| 2401 · Sole wear and alleged item swapping | Seller — incorrect | [English](docs/case-studies/case_2401/README.md) · [中文](docs/case-studies/case_2401/README.zh-CN.md) |

## 8. Baseline inputs, outputs and label use

| Baseline | Inputs and outputs | Ground-truth label use |
|---|---|---|
| **LLMs** | Provide all case text to the model in a single request to directly generate a verdict and rationale. | Ground-truth verdict labels are used only to evaluate prediction performance and are not provided as model inputs. |
| **MLLMs** | Provide all case text and multimodal evidence to the model in a single request to directly generate a verdict and rationale. | Ground-truth verdict labels are used only to evaluate prediction performance and are not provided as model inputs. |
| **AgentCourt** | The original framework supports only text inputs. Evaluation retains its text-only architecture. The original agent interaction process generates a verdict and rationale. | Ground-truth verdict labels are used only to evaluate prediction performance and are not provided as model inputs. |
| **TED** | Provide case text and multimodal evidence to multiple debating agents, collect the results of each debate round, and construct a graph from these results. Train a GAT to produce the final binary verdict. | Ground-truth verdict labels from the training set supervise GAT training. Validation and test labels are used only for performance evaluation. |
| **CyberJurors** | Provide case text and multimodal evidence to multiple juror agents, which generate a final verdict and rationale through discussion. A precedent database stores historical cases and their ground-truth verdicts. Retrieved precedents and verdicts assist reasoning about the current case. | The precedent database is constructed exclusively from the training set, and its historical labels participate in reasoning. The current validation or test case's ground-truth label is used only to evaluate prediction performance and is not provided as an input for that case. |
