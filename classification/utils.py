import torch
import random
import pandas as pd
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from transformers import NatConfig, NatModel, SwinConfig, SwinModel, DinatConfig, DinatModel, EfficientNetConfig, EfficientNetModel
from timm.models import create_model
from timm.utils import *
from timm.loss import *
from timm.scheduler import *

from nat import *
from dinat import *
from dinats import *
from isotropic import *

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomAffine(degrees = (0,50), translate=(0.1, 0.3), scale=(0.75, 0.99)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
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
        # return 320
    def open_image(self,path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = self.transform(img)
        return img
    def __getitem__(self, index):
        # return self.open_image(self.data.loc[index]['Path']), np.zeros((1)).astype('float32')
        if self.data.loc[index]['Y']== 1:
            label = np.array([0,1],dtype = 'float')
        else:
            label = np.array([1,0],dtype = 'float')

        return self.open_image(self.data.loc[index]['Path']), label.astype('float32')
        return self.open_image(self.data.loc[index]['Path']), np.expand_dims(self.data.loc[index]['Y'], -1).astype('float32')

class TestDataset(CustomDataset):
    def __init__(self,  csv, transform):
        data =pd.read_csv(csv)
        data = data.sort_values('ID')
        data['pred'] = None
        self.data = data
        self.transform = transform
        self.ID = data['ID'].unique()
    def __getitem__(self, index):
        return self.open_image(self.data.loc[index]['Path']), np.expand_dims(self.data.loc[index]['Y'], -1).astype('float32'), index
  

def pretrained_model(modelname='nat_small', num_classes = 1, freeze = False):
    model = create_model(modelname, pretrained=True)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 2) 
    if num_classes == 1:
        model.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = self.forward_features(x)
            x = self.sigmoid(self.head(x))
            return x
        model.forward =  forward.__get__(model, type(model))
    else:
        model.softmax = nn.Softmax()
        def forward(self, x):
            x = self.forward_features(x)
            x = self.softmax(self.head(x))
            return x
        model.forward =  forward.__get__(model, type(model))        
    return model

def backbone(name):
    if name == 'Nat':
        configuration = NatConfig()
        model = NatModel(configuration)
    elif name == 'DiNat':
        configuration = DinatConfig()
        model = DinatModel(configuration)
    elif name =='Swin':
        configuration = SwinConfig()
        model = SwinModel(configuration)
    elif name =='EfficientNet':
        configuration = EfficientNetConfig()
        model = EfficientNetModel(configuration)
    else:
        raise NotImplementedError('Please choose from: Nat, DiNat, Swin, EfficientNet')
    return model




class backbone_model(torch.nn.Module):
    def __init__(self, name = 'Nat',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_model = backbone(name=name)
        
    
