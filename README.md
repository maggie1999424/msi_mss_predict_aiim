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

I believe rather than random spiting, tiles from the same slide must be arranged into same train/val/test set. [DevideTrainTest.py](DevideTrainTest.py) will do this for us.

Use [DevideTrainTest.py](DevideTrainTest.py) to generate a 5-fold train/val/test set from [all_avaliable.csv](all_avaliable.csv): 

```bash
python DevideTrainTest.py -i all_avaliable.csv -R 4:1:1 #for  train/val/test = 4:1:1
```

You will get these files generated (or, just use those I've already provided):

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

### Write Config File & Run Training

I've modified [train.py](classification/train.py) from [neighborhood attention transformers](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer), by:

- changing the dataset & dataloader part from `timm` back to `torch` 
- assign loss function as `BinaryCrossEntropy` directly
- Some multi-thread things that keeps make me error

```bash
# cd into working dir
cd msi_mss_predict_aiim-main2
python classification/train.py -c ./classification/configs/MSI_small.yml
```

You must provide a config file. You may find some example in [configs](classification/configs). Many settings could be change, including different model structure.

```yaml
model: nat_small
lr: 1e-3
warmup_lr: 1e-6
min_lr: 5e-6
epochs: 12
cooldown_epochs: 2
warmup_epochs: 2
amp: True
batch_size: 32
sched: cosine
sched_epochwise: False
weight_decay: 5e-2
num_classes: 1
bce_target_thresh: 0.5
```

```
python classification/train.py --help
NO CONFIG FILE FOUND, using defaults!
usage: NAT training script [-h] [--datalist_dir datalist_dir]
                           [--datalist_name Name] [--fold_num INT]
                           [--model MODEL] [--pretrained]
                           [--initial-checkpoint PATH] [--resume PATH]
                           [--no-resume-opt] [--num-classes N] [--gp POOL]
                           [--img-size N] [--input-size N N N N N N N N N]
                           [--crop-pct N] [--mean MEAN [MEAN ...]]
                           [--std STD [STD ...]] [--interpolation NAME] [-b N]
                           [-vb N] [--disable-eval] [--opt OPTIMIZER]
                           [--opt-eps EPSILON] [--opt-betas BETA [BETA ...]]
                           [--momentum M] [--weight-decay WEIGHT_DECAY]
                           [--clip-grad NORM] [--clip-mode CLIP_MODE]
                           [--sched SCHEDULER] [--sched-epochwise] [--lr LR]
                           [--lr-noise pct, pct [pct, pct ...]]
                           [--lr-noise-pct PERCENT] [--lr-noise-std STDDEV]
                           [--lr-cycle-mul MULT] [--lr-cycle-decay MULT]
                           [--lr-cycle-limit N] [--lr-k-decay LR_K_DECAY]
                           [--warmup-lr LR] [--min-lr LR] [--epochs N]
                           [--epoch-repeats N] [--start-epoch N]
                           [--decay-epochs N] [--warmup-epochs N]
                           [--cooldown-epochs N] [--patience-epochs N]
                           [--decay-rate RATE] [--no-aug]
                           [--scale PCT [PCT ...]] [--ratio RATIO [RATIO ...]]
                           [--hflip HFLIP] [--vflip VFLIP]
                           [--color-jitter PCT] [--aa NAME]
                           [--aug-repeats AUG_REPEATS]
                           [--aug-splits AUG_SPLITS] [--jsd-loss] [--bce-loss]
                           [--bce-target-thresh BCE_TARGET_THRESH]
                           [--reprob PCT] [--remode REMODE]
                           [--recount RECOUNT] [--resplit] [--mixup MIXUP]
                           [--cutmix CUTMIX]
                           [--cutmix-minmax CUTMIX_MINMAX [CUTMIX_MINMAX ...]]
                           [--mixup-prob MIXUP_PROB]
                           [--mixup-switch-prob MIXUP_SWITCH_PROB]
                           [--mixup-mode MIXUP_MODE] [--mixup-off-epoch N]
                           [--smoothing SMOOTHING]
                           [--train-interpolation TRAIN_INTERPOLATION]
                           [--drop PCT] [--drop-connect PCT]
                           [--drop-block PCT] [--bn-tf]
                           [--bn-momentum BN_MOMENTUM] [--bn-eps BN_EPS]
                           [--sync-bn] [--dist-bn DIST_BN] [--split-bn]
                           [--model-ema] [--model-ema-force-cpu]
                           [--model-ema-decay MODEL_EMA_DECAY] [--seed S]
                           [--worker-seeding WORKER_SEEDING]
                           [--log-interval N] [--recovery-interval N]
                           [--checkpoint-hist N] [-j N] [--save-images]
                           [--amp] [--apex-amp] [--native-amp] [--no-ddp-bb]
                           [--channels-last] [--pin-mem] [--no-prefetcher]
                           [--output PATH] [--experiment NAME]
                           [--project NAME] [--eval-metric EVAL_METRIC]
                           [--tta N] [--local_rank LOCAL_RANK]
                           [--use-multi-epochs-loader] [--torchscript]
                           [--log-wandb] [--world-size WORLD_SIZE]
                           [--dist-url DIST_URL]

options:
  -h, --help            show this help message and exit
  --datalist_dir datalist_dir
                        path to dataset ListDir
  --datalist_name Name  datalist name
  --fold_num INT        which fold number to use, int
  --model MODEL         Name of model to train (default: "nat_tiny"
  --pretrained          Start with pretrained version of specified network (if
                        avail)
  --initial-checkpoint PATH
                        Initialize model from this checkpoint (default: none)
  --resume PATH         Resume full model and optimizer state from checkpoint
                        (default: none)
  --no-resume-opt       prevent resume of optimizer state when resuming model
  --num-classes N       number of label classes (Model default if None)
  --gp POOL             Global pool type, one of (fast, avg, max, avgmax,
                        avgmaxc). Model default if None.
  --img-size N          Image patch size (default: None => model default)
  --input-size N N N N N N N N N
                        Input all image dimensions (d h w, e.g. --input-size 3
                        224 224), uses model default if empty
  --crop-pct N          Input image center crop percent (for validation only)
  --mean MEAN [MEAN ...]
                        Override mean pixel value of dataset
  --std STD [STD ...]   Override std deviation of of dataset
  --interpolation NAME  Image resize interpolation type (overrides model)
  -b N, --batch-size N  input batch size for training (default: 128)
  -vb N, --validation-batch-size N
                        validation batch size override (default: None)
  --disable-eval        Disables evaluation in every epoch.
  --opt OPTIMIZER       Optimizer (default: "adamw"
  --opt-eps EPSILON     Optimizer Epsilon (default: 1e-8, use opt default)
  --opt-betas BETA [BETA ...]
                        Optimizer Betas (default: [0.9, 0.999], use opt
                        default)
  --momentum M          Optimizer momentum (default: 0.9)
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 0.05)
  --clip-grad NORM      Clip gradient norm (default: 5.0)
  --clip-mode CLIP_MODE
                        Gradient clipping mode. One of ("norm", "value",
                        "agc")
  --sched SCHEDULER     LR scheduler (default: "step"
  --sched-epochwise     Apply scheduler epochwise as opposed to iterwise.
  --lr LR               learning rate (default: 1e-3)
  --lr-noise pct, pct [pct, pct ...]
                        learning rate noise on/off epoch percentages
  --lr-noise-pct PERCENT
                        learning rate noise limit percent (default: 0.67)
  --lr-noise-std STDDEV
                        learning rate noise std-dev (default: 1.0)
  --lr-cycle-mul MULT   learning rate cycle len multiplier (default: 1.0)
  --lr-cycle-decay MULT
                        amount to decay each learning rate cycle (default:
                        1.0)
  --lr-cycle-limit N    learning rate cycle limit, cycles enabled if > 1
  --lr-k-decay LR_K_DECAY
                        learning rate k-decay for cosine/poly (default: 1.0)
  --warmup-lr LR        warmup learning rate (default: 0.000001)
  --min-lr LR           lower lr bound for cyclic schedulers that hit 0 (5e-6)
  --epochs N            number of epochs to train (default: 300)
  --epoch-repeats N     epoch repeat multiplier (number of times to repeat
                        dataset epoch per train epoch).
  --start-epoch N       manual epoch number (useful on restarts)
  --decay-epochs N      epoch interval to decay LR
  --warmup-epochs N     epochs to warmup LR, if scheduler supports
  --cooldown-epochs N   epochs to cooldown LR at min_lr, after cyclic schedule
                        ends
  --patience-epochs N   patience epochs for Plateau LR scheduler (default: 10
  --decay-rate RATE, --dr RATE
                        LR decay rate (default: 0.1)
  --no-aug              Disable all training augmentation, override other
                        train aug args
  --scale PCT [PCT ...]
                        Random resize scale (default: 0.08 1.0)
  --ratio RATIO [RATIO ...]
                        Random resize aspect ratio (default: 0.75 1.33)
  --hflip HFLIP         Horizontal flip training aug probability
  --vflip VFLIP         Vertical flip training aug probability
  --color-jitter PCT    Color jitter factor (default: 0.4)
  --aa NAME             Use AutoAugment policy. "v0" or "original". (default:
                        rand-m9-mstd0.5-inc1)
  --aug-repeats AUG_REPEATS
                        Number of augmentation repetitions (distributed
                        training only) (default: 0)
  --aug-splits AUG_SPLITS
                        Number of augmentation splits (default: 0, valid: 0 or
                        >=2)
  --jsd-loss            Enable Jensen-Shannon Divergence + CE loss. Use with
                        `--aug-splits`.
  --bce-loss            Enable BCE loss w/ Mixup/CutMix use.
  --bce-target-thresh BCE_TARGET_THRESH
                        Threshold for binarizing softened BCE targets
                        (default: None, disabled)
  --reprob PCT          Random erase prob (default: 0.25)
  --remode REMODE       Random erase mode (default: "pixel")
  --recount RECOUNT     Random erase count (default: 1)
  --resplit             Do not random erase first (clean) augmentation split
  --mixup MIXUP         mixup alpha, mixup enabled if > 0. (default: 0.8)
  --cutmix CUTMIX       cutmix alpha, cutmix enabled if > 0. (default: 1.0)
  --cutmix-minmax CUTMIX_MINMAX [CUTMIX_MINMAX ...]
                        cutmix min/max ratio, overrides alpha and enables
                        cutmix if set (default: None)
  --mixup-prob MIXUP_PROB
                        Probability of performing mixup or cutmix when
                        either/both is enabled
  --mixup-switch-prob MIXUP_SWITCH_PROB
                        Probability of switching to cutmix when both mixup and
                        cutmix enabled
  --mixup-mode MIXUP_MODE
                        How to apply mixup/cutmix params. Per "batch", "pair",
                        or "elem"
  --mixup-off-epoch N   Turn off mixup after this epoch, disabled if 0
                        (default: 0)
  --smoothing SMOOTHING
                        Label smoothing (default: 0.1)
  --train-interpolation TRAIN_INTERPOLATION
                        Training interpolation (random, bilinear, bicubic
                        default: "random")
  --drop PCT            Dropout rate (default: 0.)
  --drop-connect PCT    Drop connect rate, DEPRECATED, use drop-path (default:
                        None)
  --drop-block PCT      Drop block rate (default: None)
  --bn-tf               Use Tensorflow BatchNorm defaults for models that
                        support it (default: False)
  --bn-momentum BN_MOMENTUM
                        BatchNorm momentum override (if not None)
  --bn-eps BN_EPS       BatchNorm epsilon override (if not None)
  --sync-bn             Enable NVIDIA Apex or Torch synchronized BatchNorm.
  --dist-bn DIST_BN     Distribute BatchNorm stats between nodes after each
                        epoch ("broadcast", "reduce", or "")
  --split-bn            Enable separate BN layers per augmentation split.
  --model-ema           Enable tracking moving average of model weights
  --model-ema-force-cpu
                        Force ema to be tracked on CPU, rank=0 node only.
                        Disables EMA validation.
  --model-ema-decay MODEL_EMA_DECAY
                        decay factor for model weights moving average
                        (default: 0.9998)
  --seed S              random seed (default: 42)
  --worker-seeding WORKER_SEEDING
                        worker seed mode (default: all)
  --log-interval N      how many batches to wait before logging training
                        status
  --recovery-interval N
                        how many batches to wait before writing recovery
                        checkpoint
  --checkpoint-hist N   number of checkpoints to keep (default: 10)
  -j N, --workers N     how many training processes to use (default: 8)
  --save-images         save images of input bathes every log interval for
                        debugging
  --amp                 use NVIDIA Apex AMP or Native AMP for mixed precision
                        training
  --apex-amp            Use NVIDIA Apex AMP mixed precision
  --native-amp          Use Native Torch AMP mixed precision
  --no-ddp-bb           Force broadcast buffers for native DDP to off.
  --channels-last       Use channels_last memory layout
  --pin-mem             Pin CPU memory in DataLoader for more efficient
                        (sometimes) transfer to GPU.
  --no-prefetcher       disable fast prefetcher
  --output PATH         path to output folder (default: none, current dir)
  --experiment NAME     name of train experiment, name of sub-folder for
                        output
  --project NAME        project name
  --eval-metric EVAL_METRIC
                        Best metric (default: "top1"
  --tta N               Test/inference time augmentation (oversampling)
                        factor. 0=None (default: 0)
  --local_rank LOCAL_RANK
  --use-multi-epochs-loader
                        use the multi-epochs-loader to save time at the
                        beginning of every epoch
  --torchscript         convert model torchscript for inference
  --log-wandb           log training and validation metrics to wandb
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training
```

**(Some of the arguments are disabled by me)**

Try more hyperparameters plz!

Currently the code works on 1 GPU only. Still trying a more efficient way......

### Save Path

The best weight `.pth` and the epoch summary will be automatically saved in [output](output)
