import torch
import random
import pandas as pd
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from transformers import NatConfig, NatModel
import transformers
# from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
        model_parameters
    # convert_splitbn_model,
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import *
from timm.utils import ApexScaler, NativeScaler

from nat import *
from dinat import *
from dinats import *
from isotropic import *

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class CustomDataset(Dataset):
    def __init__(self, csv, transform):
        self.data = pd.read_csv(csv)
        self.transform = transform
    def __len__(self):
        return len(self.data)
        # return 8000
    def open_image(self,path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = self.transform(img)
        return img
    def __getitem__(self, index):
        # return self.open_image(self.data.loc[index]['Path']), np.zeros((1)).astype('float32')
        return self.open_image(self.data.loc[index]['Path']), np.expand_dims(self.data.loc[index]['Y'], -1).astype('float32')


def pretrained_model(modelname='nat_small', num_classes = 1, freeze = False):
    model = create_model(modelname, pretrained=True)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)    
    if num_classes == 1:
        model.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = self.forward_features(x)
            x = self.sigmoid(self.head(x))
            return x
        model.forward =  forward.__get__(model, type(model))
    return model

def backbone():
    configuration = NatConfig()
    model = NatModel(configuration)
    return model
