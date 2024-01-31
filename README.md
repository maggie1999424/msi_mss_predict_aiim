# MSI/MSS Histology Classification

This is a team project for MSI/MSS histology image classification, by using [neighborhood attention transformers](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) model. Most code is modified from the repository.

## Data

Download your data from [Kaggle: TCGA COAD MSI vs MSS Prediction (JPG)](https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg/)

Unzip and place data folders as such structure. It must be 6 GB in total:

```
kaggle-MSI
├── MSIMUT_JPEG
└── MSS_JPEG
```

File [all_avaliable.csv](all_avaliable.csv) listed all the jpeg images tiles by their original slides.

 [TCGA-COAD_MSS.tsv](TCGA-COAD_MSS.tsv)  and  [TCGA-COAD_MSI.tsv](TCGA-COAD_MSI.tsv) are the original MSI status label of each slide from TCGA-COAD official website.

## ENV Requirements

I use Ubuntu 22.04, with CUDA 11.8 and torch==2.0.1+cu118.

For other packages you may see [requirements.txt](requirements.txt) .

Note that [NATTEN](https://github.com/SHI-Labs/NATTEN) must be installed from the wheel corresponds to your CUDA and Torch version. Please refers to the explanation of NATTEN official below:

"""

Just refer to our website, [shi-labs.com/natten](https://www.shi-labs.com/natten/), select your PyTorch version and the CUDA version it was compiled with, copy-paste the command and install in seconds!

For example, if you're on `torch==2.0.0+cu118`, you should install NATTEN using the following wheel:

```bash
pip3 install natten -f https://shi-labs.com/natten/wheels/cu118/torch2.0.0/index.html
```

"""

## Running Codes

### Train/Val/Test Split

但是我弄丟了

```
List/
├── Test_list_Raw
├── Train_list_0_Raw
├── Train_list_1_Raw
├── Train_list_2_Raw
├── Train_list_3_Raw
├── Train_list_4_Raw
├── Val_list_0_Raw
├── Val_list_1_Raw
├── Val_list_2_Raw
├── Val_list_3_Raw
└── Val_list_4_Raw
```

### Run Training

see  [huggingface-with-pretrain.ipynb](huggingface-with-pretrain.ipynb) 
