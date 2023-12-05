#!python3
'''
This script is modified from
https://github.com/SHI-Labs/Neighborhood-Attention-Transformer
'''

import argparse
import time
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.utils.data import DataLoader

# from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
     model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import *
from timm.utils import ApexScaler, NativeScaler

from nat import *
from dinat import *
from dinats import *
from isotropic import *
from extras import get_gflops, get_mparams
from utils import CustomDataset, transform

import yaml
import builtins as __builtin__
builtin_print = __builtin__.print
import csv



def get_args_parser(parents=[], read_config=False):
    if read_config:
        # The first arg parser parses out only the --config argument, this argument is used to
        # load a yaml file containing key-values that override the defaults for the main parser below
        config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
        parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                            help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser('NAT training script', parents=parents)

    # Dataset / Model parameters
    # parser.add_argument('data_dir', metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')
    parser.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')

    # Model parameters
    parser.add_argument('--model', default='nat_tiny', type=str, metavar='MODEL',
                        help='Name of model to train (default: "nat_tiny"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=224, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--crop-pct', default=0.875, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='validation batch size override (default: None)')
    parser.add_argument('--disable-eval', action='store_true', default=False,
                        help='Disables evaluation in every epoch.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8, use opt default)')
    parser.add_argument('--opt-betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.999], use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip-grad', type=float, default=5.0, metavar='NORM',
                        help='Clip gradient norm (default: 5.0)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--sched-epochwise', action='store_true', default=False,
                        help='Apply scheduler epochwise as opposed to iterwise.')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=1.0, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.000001, metavar='LR',
                        help='warmup learning rate (default: 0.000001)')
    parser.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--aug-repeats', type=int, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--worker-seeding', type=str, default='all',
                        help='worker seed mode (default: all)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint-hist', type=int, default=4, metavar='N',
                        help='number of checkpoints to keep (default: 10)')
    parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                        help='how many training processes to use (default: 8)')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--project', default='', type=str, metavar='NAME',
                        help='project name')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    if read_config:
        return parser, config_parser
    return parser


def _parse_args(read_config=False):
    setup_default_logging()
    parser = get_args_parser(read_config=read_config)
    if not read_config:
        return parser.parse_args(), parser
    parser, config_parser = parser
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config and os.path.isfile(args_config.config):
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    else:
        builtin_print("NO CONFIG FILE FOUND, using defaults!")

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args, parser

def create_scheduler(args, optimizer, n_samples):
    iters_per_epoch = n_samples // (args.batch_size * args.world_size)
    num_epochs = args.epochs
    num_steps = num_epochs * iters_per_epoch
    warmup_steps = args.warmup_epochs * iters_per_epoch
    patience_steps = args.patience_epochs * iters_per_epoch
    decay_steps = args.decay_epochs * iters_per_epoch

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(args, 'lr_noise_pct', 0.67),
        noise_std=getattr(args, 'lr_noise_std', 1.),
        noise_seed=getattr(args, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(args, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(args, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(args, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs if args.sched_epochwise else num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs if args.sched_epochwise else warmup_steps,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            t_in_epochs=args.sched_epochwise,
            **cycle_args,
            **noise_args,
        )
        cycle_length = lr_scheduler.get_cycle_length() if args.sched_epochwise else lr_scheduler.get_cycle_length() // iters_per_epoch
        num_epochs = cycle_length + args.cooldown_epochs
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs if args.sched_epochwise else num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs if args.sched_epochwise else warmup_steps,
            t_in_epochs=args.sched_epochwise,
            **cycle_args,
            **noise_args,
        )
        cycle_length = lr_scheduler.get_cycle_length() if args.sched_epochwise else lr_scheduler.get_cycle_length() // iters_per_epoch
        num_epochs = cycle_length + args.cooldown_epochs
    elif args.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs if args.sched_epochwise else decay_steps,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs if args.sched_epochwise else warmup_steps,
            t_in_epochs=args.sched_epochwise,
            **noise_args,
        )
    elif args.sched == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs if args.sched_epochwise else decay_steps,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs if args.sched_epochwise else warmup_steps,
            t_in_epochs=args.sched_epochwise,
            **noise_args,
        )
    elif args.sched == 'plateau':
        mode = 'min' if 'loss' in getattr(args, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_epochs if args.sched_epochwise else patience_steps,
            lr_min=args.min_lr,
            mode=mode,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs if args.sched_epochwise else warmup_steps,
            cooldown_t=0,
            **noise_args,
        )
    elif args.sched == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=args.decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=num_epochs if args.sched_epochwise else num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs if args.sched_epochwise else warmup_steps,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            t_in_epochs=args.sched_epochwise,
            **cycle_args,
            **noise_args,
        )
        cycle_length = lr_scheduler.get_cycle_length() if args.sched_epochwise else lr_scheduler.get_cycle_length() // iters_per_epoch
        num_epochs = cycle_length + args.cooldown_epochs

    return lr_scheduler, num_epochs

def main(args):

    builtin_print('Training with a single process on 1 GPUs.')
    # resolve AMP arguments based on PyTorch / Apex availability


    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        # drop_rate=args.drop,
        # drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        # drop_path_rate=args.drop_path,
        # drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    model.cuda()

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0

    # move model to GPU, enable channels last layout if set
    model.cuda()

    # # setup synchronized BatchNorm for distributed training
    # if args.distributed and args.sync_bn:
    #     assert not args.split_bn
    #     if has_apex and use_amp == 'apex':
    #         # Apex SyncBN preferred unless native amp is activated
    #         model = convert_syncbn_model(model)
    #     else:
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     if args.local_rank == 0:
    #         builtin_print(
    #             'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
    #             'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    # if args.torchscript:
    #     assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
    #     assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
    #     model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # # setup automatic mixed-precision (AMP) loss scaling and op casting
    # amp_autocast = suppress  # do nothing
    # loss_scaler = None
    # if use_amp == 'apex':
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    #     loss_scaler = ApexScaler()
    #     if args.local_rank == 0:
    #         builtin_print('Using NVIDIA APEX AMP. Training in mixed precision.')
    # elif use_amp == 'native':
    #     amp_autocast = torch.cuda.amp.autocast
    #     loss_scaler = NativeScaler()
    #     if args.local_rank == 0:
    #         builtin_print('Using native Torch AMP. Training in mixed precision.')
    # else:
    #     if args.local_rank == 0:
    #         builtin_print('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None

    custom_dataset = CustomDataset('all_avaliable.csv',transform)
    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer, len(custom_dataset))
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        builtin_print('Scheduled epochs: {}'.format(num_epochs))

    loader_train = DataLoader(dataset=custom_dataset, 
                            batch_size=args.batch_size,
                              shuffle=True)

    loader_eval = None

    train_loss_fn =torch.nn.BCELoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None

    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=None, loss_scaler=None, model_ema=model_ema, mixup_fn=None)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    builtin_print("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
            if loader_eval is not None:
                eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                    ema_eval_metrics = validate(
                        model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if lr_scheduler is not None and args.sched_epochwise:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, None if eval_metrics is None else eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = None if eval_metrics is None else eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        builtin_print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    num_iters = len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward(create_graph=second_order)
        optimizer.step()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                builtin_print(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None and args.sched_epochwise:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for
    return OrderedDict([('loss', losses_m.avg)])


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


if __name__ == '__main__':
    args, _ = _parse_args(read_config=True)
    main(args)
