# Temporal Evidence-Aware Multi-Agent Reasoning for E-Commerce Dispute Verdict

[English](README.md) | 简体中文 ![README 访问量](https://komarev.com/ghpvc/?username=Chronojury&label=README+%E8%AE%BF%E9%97%AE%E9%87%8F&color=blue&style=flat)

ChronoJury 通过两个模块处理按时间顺序出现的纠纷证据：

- **SWER — 阶段式证据推理：** 将证据组织为 `T` 个阶段，检索和感知证据，并生成各阶段的陪审智能体裁决。智能体通过有向关注网络交换意见。
- **STVA — 时空裁决聚合：** 编码智能体裁决和阶段案件摘要，聚合图信息，通过因果时序注意力建模意见演变，生成最终裁决。

## 1. 仓库结构

```text
chronojury/
  swer/                   # 证据推理与陪审智能体交互
  stva/                   # 特征编码与裁决聚合
  explanation/            # GNNExplainer 报告与案例分析
  llm.py                  # 模型接口路由
  runtime.py              # 并发管理
  tracing.py              # 推理中间记录
run.sh                    # 流程入口
requirements.txt          # 环境依赖
pyproject.toml            # 包安装配置
```

## 2. 安装

环境使用 Python 3.10、PyTorch 2.7.1 和 CUDA 12.8。服务器需安装支持 CUDA 12.8 的 NVIDIA 驱动。在仓库根目录执行：

```bash
conda create -n chronojury python=3.10 pip -y
conda activate chronojury
python -m pip install -r requirements.txt -e .
```

Conda 创建环境，pip 一次安装依赖和项目。

## 3. 数据集：VerdictBench

ChronoJury 使用 Hugging Face 上的 [VerdictBench](https://huggingface.co/datasets/piggi/VerdictBench)，该数据集由 [CyberJurors（Sun 等，2026）](https://arxiv.org/abs/2605.28369) 提出。它包含五类商品的 6,000 起真实电商纠纷，提供交易信息、买卖双方主张，以及多轮文本、图片和视频证据。数据集依据商品类别和投票差距分层，按约 3:1:2 划分训练、验证和测试集。

[VerdictBench](https://huggingface.co/datasets/piggi/VerdictBench) 要求申请访问权限，并标明许可为 **CC BY-NC 4.0**、用途为学术研究。

### 3.1 标签与参考裁决

训练和评估直接使用 `case_data.whowins`，不重新标注：

| 数值 | 含义 |
|---|---|
| `0` | 平台裁决支持买家。 |
| `1` | 平台裁决支持卖家。 |

VerdictBench（ICML 2026）基于真实世界中的纠纷构建。每起案件均由人类陪审员审理，获得更多票数的一方被判定为胜诉方。如果买方胜诉，平台会执行退款。否则，交易完成。如果任一纠纷方对裁决提出异议，平台会安排二次复核。在这一机制下，投票差距可以反映纠纷的裁决难度。例如，9:8 的投票结果表明双方均有得到陪审员支持的合理主张，因此该案件更难裁决。所有裁决均在最终决定作出后才被收集，以确保其反映公平且正确的结果。

```bibtex
@article{sun2026cyberjurors,
  title={CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict},
  author={Sun, Yanhui and Liu, Wu and Ming, Haifeng and Wang, Xinru and Yao, Hantao and Zhang, Yongdong},
  journal={arXiv preprint arXiv:2605.28369},
  year={2026}
}
```

### 3.2 本地文件组织

设置本地根目录：

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

每个 `data.json` 保留 VerdictBench 的 `id`、案件元信息、`buyer_claim`、`seller_claim`、`case_details.evidences` 和 `whowins`。证据条目保留 `identity`（`买家` 或 `卖家`）及 `content` 中的 `text`、`image_paths`、`video_paths` 等字段。

### 3.3 二元裁决的适用范围

VerdictBench 不包含部分退款或双方共同承担责任的裁决案例，这与数据来源平台的实际纠纷处理机制一致：平台最终裁决仅判定买方或卖方胜诉。若双方自行达成和解，例如部分退款或共同分担责任，纠纷通常会被撤销，不产生最终裁决，因此不属于本研究范围。

## 4. 模型接口

所有大模型调用均通过兼容 OpenAI 的 chat-completions 接口完成。选择下面一种方式，并将 `CHRONOJURY_MODEL` 设为接口实际提供的多模态模型名称。

### 4.1 托管 API

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_BASE_URL='https://api.example.com/v1'
export CHRONOJURY_MODEL='your-vision-model'
```

### 4.2 自部署模型

支持并发调用部署在多张 GPU 上的模型副本。将 IP 和端口占位符替换为服务地址，各副本须提供相同模型。

```bash
export OPENAI_API_KEY='EMPTY'
export CHRONOJURY_API_BASES='http://<SERVER_IP_1>:<PORT_1>/v1,http://<SERVER_IP_2>:<PORT_2>/v1'
export CHRONOJURY_MODEL='your_model_name'
```

只有服务不要求 API 密钥认证时才使用 `EMPTY`。`CHRONOJURY_API_BASES` 优先于 `OPENAI_BASE_URL`，执行 `unset CHRONOJURY_API_BASES` 可切回单个接口。客户端支持并发请求、CPU 数据预取，并暂时避开失败的副本。

## 5. 运行流程

先激活环境，准备数据和本地编码器权重，并配置模型接口。在仓库根目录执行：

```bash
bash run.sh reason       # 生成各阶段智能体裁决
bash run.sh summarize    # 生成阶段案件摘要
bash run.sh encode       # 编码智能体和案件节点
bash run.sh train        # 训练 STVA
bash run.sh test         # 评估已保存的检查点
bash run.sh help         # 查看使用说明
```

也可以使用 `bash run.sh all` 依次运行这五步。结果保存在 `$CHRONOJURY_LOCAL_ROOT/results/` 下。推理、摘要和编码会处理训练、验证和测试划分。推理和摘要默认跳过已有结果。

各阶段的案件摘要使用截至该阶段已出现的证据，以及案件元信息和买卖双方主张，不包含后续阶段证据。智能体接收自身历史意见及所关注邻居的原始意见。

默认编码器为 `chinese-roberta-wwm-ext-large`，生成 1024 维特征。也支持 Qwen 编码，通过 `CHRONOJURY_ENCODER_BACKEND=qwen` 选择，使用 `CHRONOJURY_ENCODER_PATH` 指定本地权重，并将 `CHRONOJURY_INPUT_DIM` 设为编码器的输出维度。推理模型与文本编码器独立配置。

训练、评估和解释须保持编码器、图及模型配置一致。参数集中在 [SWER 配置](chronojury/swer/config.py)和 [STVA 配置](chronojury/stva/config.py)中维护。

## 6. GNNExplainer 报告

报告从**阶段内的信息聚合**和**跨阶段的信息传递**两个层面解释裁决，使用已训练的模型，无需重新训练。

```bash
bash run.sh explain --case-id 1            # 生成空间与时间解释报告
bash run.sh explain --case-id 1 --summary  # 同时生成辅助文字总结
```

报告从模型的实际计算过程提取解释：

1. **读取输入：** 加载训练好的模型检查点、案件各阶段的节点特征及对应的关注网络，计算模型的原始裁决。
2. **学习消息重要性：** 冻结模型参数，以模型自身的预测为解释目标，同时学习空间边和跨阶段时间消息的掩码，不使用真实标签。
3. **定位关键连接：** 提取各阶段的空间关键路径，以及同一节点从较早阶段传向较晚阶段的重要时间连接。
4. **生成报告：** 结合关键连接、相关评审员的原始意见及案件摘要，说明裁决依据和跨阶段的信息传递。

结构化报告无需再次调用大模型。可追加 `--summary` 生成辅助文字总结，该总结不参与裁决。路径重要性用于解释模型的预测，不构成因果证明。

## 7. 案例分析

三个案例包含争议证据、评审员社交关系、每位评审员三轮的裁决与理由，以及 GNNExplainer 的空间与时间解释。

| 案例 | 模型裁决 | 分析报告 |
|---|---|---|
| 702 · 手表描述不符纠纷 | 买家，预测正确 | [中文](docs/case-studies/case_702/README.zh-CN.md) · [English](docs/case-studies/case_702/README.md) |
| 2588 · 二手书破损披露纠纷 | 卖家，预测正确 | [中文](docs/case-studies/case_2588/README.zh-CN.md) · [English](docs/case-studies/case_2588/README.md) |
| 2401 · 鞋底磨损与调包争议 | 卖家，预测错误 | [中文](docs/case-studies/case_2401/README.zh-CN.md) · [English](docs/case-studies/case_2401/README.md) |

## 8. 基线输入、输出及标签使用

| 基线 | 输入与输出 | 真实标签使用 |
|---|---|---|
| **LLMs** | 将全部文本案例信息一次性输入模型，直接生成裁决结果和理由。 | 真实裁决标签仅用于评估模型预测性能，不作为模型输入。 |
| **MLLMs** | 将全部文本案例信息及多模态证据一次性输入模型，直接生成裁决结果和理由。 | 真实裁决标签仅用于评估模型预测性能，不作为模型输入。 |
| **AgentCourt** | 原模型框架仅支持文本输入，测试时保留其原有纯文本架构。通过原有智能体交互流程生成裁决结果和理由。 | 真实裁决标签仅用于评估模型预测性能，不作为模型输入。 |
| **TED** | 将案例文本及多模态证据输入多个辩论智能体，通过多轮辩论获得各轮结果，并据此构建图网络。经 GAT 训练生成最终的二元分类裁决结果。 | 训练集真实裁决标签用于 GAT 的监督训练。验证集和测试集标签仅用于模型性能评估。 |
| **CyberJurors** | 将案例文本及多模态证据输入多个陪审员智能体，通过多智能体讨论生成最终裁决结果及理由。先例库存储历史案例及其真实裁决标签，检索到的先例和裁决结果辅助当前案例的推理与判断。 | 先例库仅由训练集构建，其中的历史案例标签会参与推理。当前验证或测试案例的真实标签仅用于评估预测性能，不作为当前案例的输入。 |
