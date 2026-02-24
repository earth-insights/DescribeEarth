# DescribeEarth: Describe Anything for Remote Sensing Images

Author: [Kaiyu Li](https://likyoo.github.io/)\*, [Zixuan Jiang](https://anxmuy.github.io/)\*, Xiangyong Cao✉, Jiayu Wang, Yuchen Xiao, Jing Yao, Chen Wu, Deyu Meng, Zhi Wang

---

## News

- **`2025-10-01`**: 🔥🔥🔥The [paper](https://arxiv.org/abs/2509.25654), [code](https://github.com/earth-insights/DescribeEarth), [dataset](https://huggingface.co/datasets/earth-insights/DE-Dataset) and [benchmark](https://huggingface.co/datasets/earth-insights/DE-Benchmark) are released.
---

## Introduction

![](https://github.com/user-attachments/assets/bcfe50ae-945b-448f-aaba-c7a02bb96c80)

Automated textual description of remote sensing images is essential for applications such as environmental monitoring, urban planning, and disaster management. However, most existing methods only generate captions at the image level, lacking fine-grained object-level interpretation.

To address this gap, we propose **Geo-DLC**, a new task of **object-level fine-grained image captioning** for remote sensing. To support this task, we introduce:

- **DE-Dataset**: a large-scale dataset with 25 categories and 261,806 annotated instances, providing detailed descriptions of object attributes, relationships, and contexts.
- **DE-Benchmark**: an LLM-assisted question-answering evaluation suite to systematically measure model performance on Geo-DLC.
- **DescribeEarth**: a Multi-modal Large Language Model (MLLM) explicitly designed for Geo-DLC, featuring a scale-adaptive focal strategy and a domain-guided fusion module to capture both high-resolution details and global context.

**DescribeEarth** consistently outperforms state-of-the-art general MLLMs on DE-Benchmark, achieving superior factual accuracy, descriptive richness, and grammatical soundness across diverse remote sensing scenarios.

---


## Installation

see [here](environments/README.md)

## Usage

### DEMO

```python
cd scripts
python app.py
```

### De-Dataset & De-Benchmark

De-Dataset can be downloaded from [here](https://huggingface.co/datasets/earth-insights/DE-Dataset). The dataset is formatted in the form as follow:

```yaml
DE-Dataset
- {DIOR, DOTA}
- - image
- - description
```

Use `bash scripts/format_data.sh` to format data for training.  

De-Benchmark can be downloaded from [huggingface](https://huggingface.co/datasets/earth-insights/DE-Benchmark).

### Quick Start

the Pretrained checkpoints of DescribeEarth can be downloaded from [huggingface](https://huggingface.co/earth-insights/DescribeEarth). To use it, put the whole folder in `weights/`.

### Inference

```sh
python inference.py --model_dir <model_dir> --image <image_dir> --bbox <4-points-bbox/2-points-bbox>
```

- **Example**

    ````sh
    python inference.py --model_dir ../weights/DescribeEarth_0930 --image ./example1/image.jpg --bbox 36.0 332.0 311.0 325.0 317.0 584.0 42.0 591.0
    ````

- **Result**

    `````tex
    The object of category baseball_field within the specified polygon bounding box is a well-defined outdoor sports facility designed for baseball. The field features a central dirt infield area, clearly demarcated from the surrounding grassy outfield. The infield includes a pitcher's mound and bases, indicating its purpose for baseball games. The surrounding area consists of a large, open grassy field, typical of a baseball diamond layout. Adjacent to the field are structures that appear to be part of a larger complex, possibly including facilities such as dugouts or storage areas. The overall layout and design confirm this as a dedicated baseball field. There are no visible signs of current activity on the field itself.
    `````

### Training

Following Qwen2.5-VL baseline, do the following to train on DE-dataset / your own dataset:

1. Edit `Qwen2.5-VL/qwen-vl-finetune/qwenvl/data/__init__.py` for the Path to the Formatted dataset.
2. Download pretrained weights (merged checkpoint of Qwen2.5-VL-3B and RemoteCLIP-vit-b32) from [huggingface](https://huggingface.co/earth-insights/Qwen2.5-VL-3B-RC-1120).
3. `bash script/sft.sh` under `Qwen2.5-VL/qwen-vl-finetune`

### Evaluating

Use `scripts/openai_valid.py` to evaluate DescribeEarth and other models.

`````sh
python openai_valid.py <path to QA.json> <path to image_dataset> -o <output_dir> --generator <'api' or 'local'> --api-key <api_key> --model_dir <model_dir>
`````


Use `calculate_score.py` to get the final results. 

## BibTeX

```bibtex
@article{li2025describeearth,
  title={DescribeEarth: Describe Anything for Remote Sensing Images},
  author={Li, Kaiyu and Jiang, Zixuan and Cao, Xiangyong and Wang, Jiayu and Xiao Yuchen and Meng, Deyu and Wang, Zhi},
  journal={arXiv preprint arXiv:2509.25654},
  year={2025}
}
```
