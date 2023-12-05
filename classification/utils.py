import torch
import random
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class CustomDataset(Dataset):
    def __init__(self, csv, transform):
        self.data = pd.read_csv(csv)
        self.transform = transform
    def __len__(self):
        return len(self.data)
        # return 300
    def open_image(self,path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = self.transform(img)
        return img
    def __getitem__(self, index):
        return self.open_image(self.data.loc[index]['Path']), np.expand_dims(self.data.loc[index]['Y'], -1).astype('float32')

def get_dataset(listdir,name, fold_num= 0, transform = transform):
    train_dataset = CustomDataset(os.path.join(listdir,f'Train_list_{fold_num}_{name}'), transform)
    val_dataset = CustomDataset(os.path.join(listdir,f'Val_list_{fold_num}_{name}'), transform)
    return train_dataset, val_dataset



