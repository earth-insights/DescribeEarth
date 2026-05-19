<div align="center">

# DescribeEarth

### Describe Anything for Remote Sensing Images

<p>
  <a href="https://arxiv.org/abs/2509.25654"><img src="https://img.shields.io/badge/arXiv-2509.25654-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/earth-insights/DescribeEarth"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow.svg" alt="Model"></a>
  <a href="https://huggingface.co/datasets/earth-insights/DE-Dataset"><img src="https://img.shields.io/badge/Dataset-DE--Dataset-blue.svg" alt="Dataset"></a>
  <a href="https://huggingface.co/datasets/earth-insights/DE-Benchmark"><img src="https://img.shields.io/badge/Benchmark-DE--Benchmark-purple.svg" alt="Benchmark"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

Author: [Kaiyu Li](https://likyoo.github.io/)\*, [Zixuan Jiang](https://anxmuy.github.io/)\*, Xiangyong Cao✉, Jiayu Wang, Yuchen Xiao, Jing Yao, Chen Wu, Deyu Meng, Zhi Wang

</div>

---

## Overview

**DescribeEarth** is a remote-sensing MLLM framework for **object-level fine-grained image description**, rather than only image-level captioning.  
It targets practical Earth observation scenarios such as environmental monitoring, urban planning, and disaster analysis.

We introduce the **Geo-DLC** task and provide a complete research stack:

- **DE-Dataset**: a large-scale dataset with 25 categories and 261,806 object instances.
- **DE-Benchmark**: an LLM-assisted QA benchmark for factuality, richness, and language quality.
- **DescribeEarth model**: a domain-aware MLLM with scale-adaptive and fusion strategies for remote-sensing imagery.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bcfe50ae-945b-448f-aaba-c7a02bb96c80" width="95%" alt="DescribeEarth Framework"/>
</p>

---

## News

- **2025-10-01**: Paper, code, dataset, benchmark, and checkpoints released.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [Citation](#citation)

---

## Installation

Please follow environment setup instructions in [environments/README.md](environments/README.md).

---

## Quick Start

### 1) Download model weights

Download pretrained checkpoints from [Hugging Face](https://huggingface.co/earth-insights/DescribeEarth), then place them under:

```text
weights/
└── DescribeEarth_xxx
```

### 2) Run inference

```bash
python scripts/inference.py \
  --model_dir <model_dir> \
  --image <image_path> \
  --bbox <4-point bbox or 2-point bbox>
```

Example:

```bash
python scripts/inference.py \
  --model_dir ./weights/DescribeEarth_0930 \
  --image ./example1/image.jpg \
  --bbox 36.0 332.0 311.0 325.0 317.0 584.0 42.0 591.0
```

Example output:

```text
The object of category baseball_field within the specified polygon bounding box is a well-defined outdoor sports facility designed for baseball...
```

---

## Data Preparation

### DE-Dataset

Download from [DE-Dataset on Hugging Face](https://huggingface.co/datasets/earth-insights/DE-Dataset).

Expected structure:

```yaml
DE-Dataset
- {DIOR, DOTA}
- - image
- - description
```

Format data for training:

```bash
bash scripts/format_data.sh
```

### DE-Benchmark

Download from [DE-Benchmark on Hugging Face](https://huggingface.co/datasets/earth-insights/DE-Benchmark).

---

## Training

Following the Qwen2.5-VL baseline, train on DE-Dataset (or your own data) with these steps:

1. Set formatted dataset path in `DescribeEarth_Qwen2.5-VL/qwen-vl-finetune/qwenvl/data/__init__.py`.
2. Download merged pretrained weights from [Qwen2.5-VL-3B-RC-1120](https://huggingface.co/earth-insights/Qwen2.5-VL-3B-RC-1120).
3. Run:

```bash
cd DescribeEarth_Qwen2.5-VL/qwen-vl-finetune
bash scripts/sft.sh
```

---

## Evaluation

Use `scripts/openai_valid.py` to evaluate DescribeEarth or other models:

```bash
python scripts/openai_valid.py \
  <path_to_QA.json> \
  <path_to_image_dataset> \
  -o <output_dir> \
  --generator <api|local> \
  --api-key <api_key> \
  --model_dir <model_dir>
```

Then compute final scores:

```bash
python scripts/calculate_score.py
```

---

## Demo

Launch the local demo app:

```bash
cd scripts
python app.py
```

---

## Citation

If you find this project useful, please cite:

```bibtex
@article{li2025describeearth,
  title={DescribeEarth: Describe Anything for Remote Sensing Images},
  author={Li, Kaiyu and Jiang, Zixuan and Cao, Xiangyong and Wang, Jiayu and Xiao, Yuchen and Wu, Chen and Meng, Deyu and Wang, Zhi},
  journal={arXiv preprint arXiv:2509.25654},
  year={2025}
}
```

---
