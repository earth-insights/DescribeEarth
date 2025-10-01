# 🚀 DescribeEarth Environment Setup

## 1. Create and Activate Environment
```bash
conda create -n DescribeEarth python=3.10 -y
conda activate DescribeEarth
```
## 2. Install Dependencies
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm==1.0.19
pip install transformers==4.50.0
pip install qwen-vl-utils==0.0.11
pip install pillow==11.0.0
pip install datasets==3.6.0
pip install flash_attn==2.7.4.post1
pip install deepspeed==0.16.4
pip install triton==3.3.0
pip install accelerate==1.4.0
pip install torchcodec==0.2
pip install open_clip_torch
pip install decord==0.6.0
pip install openpyxl
```
