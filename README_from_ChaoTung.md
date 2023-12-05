# Image Classification

## Requirements
Python 3.8 or higher is recommended.
PyTorch version `1.11` is recommended for reproducibility, but you shouldn't face any issues going all the way up to 1.13.
For ease of use, you can just set up a new environment and run the following:
```shell
pip3 install -r requirements-base.txt # Installs torch
pip3 install -r requirements.txt # Installs NATTEN, timm, and fvcore
```
This will install the recommended torch and torchvision, 
our PyTorch extension ([NATTEN](https://github.com/SHI-Labs/NATTEN)),
[timm](https://github.com/rwightman/pytorch-image-models/), 
and all other dependencies.

Our models are based on PyTorch, and was trained on ImageNet-1K classification using the 
[timm](https://github.com/rwightman/pytorch-image-models/) package. 
Additionally, they depend on our extension, [NATTEN](https://github.com/SHI-Labs/NATTEN), which you can install 
[by referring to our website](https://www.shi-labs.com/natten/). 
The version of the [timm](https://github.com/rwightman/pytorch-image-models/) training script available here also 
requires [https://github.com/facebookresearch/fvcore](fvcore) to count FLOPs.
The following are the recommended versions of these libraries and are strongly encouraged for reproducibility and speed:
```shell
torch==1.11.0+cu113
torchvision==0.12.0+cu113

natten==0.14.4+torch111cu113 # Wheels: http://www.shi-labs.com/natten/wheels/cu113/torch1.11/index.html

timm==0.5.0
fvcore==0.1.5.post20220305
pyyaml==6.0
```

## Models

* [Neighborhood Attention Transformer (NAT)](NAT.md)

* [Dilated Neighborhood Attention Transformer (DiNAT)](DiNAT.md)



## Install

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Just refer to our website, [shi-labs.com/natten](https://www.shi-labs.com/natten/), select your PyTorch version and the CUDA version it was compiled with, copy-paste the command and install in seconds!

For example, if you're on `torch==2.0.0+cu118`, you should install NATTEN using the following wheel:

```bash
pip3 install natten -f https://shi-labs.com/natten/wheels/cu118/torch2.0.0/index.html
```
