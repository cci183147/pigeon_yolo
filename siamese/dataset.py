# siamese/dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

DEFAULT_SIZE = 224

def get_transforms(train=True, size=DEFAULT_SIZE):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8,1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1,0.1,0.1,0.05),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize((size,size)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

class SiamesePairsDataset(Dataset):
    def __init__(self, pairs_csv, crops_root, transform=None):
        self.df = pd.read_csv(pairs_csv)
        self.root = crops_root
        self.transform = transform if transform is not None else get_transforms(train=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a_path = row['img1']
        b_path = row['img2']
        label = int(row['label'])
        # if paths are relative to crops_root or absolute, handle:
        a_full = a_path if os.path.isabs(a_path) else os.path.join(self.root, a_path)
        b_full = b_path if os.path.isabs(b_path) else os.path.join(self.root, b_path)
        a = Image.open(a_full).convert('RGB')
        b = Image.open(b_full).convert('RGB')
        a = self.transform(a)
        b = self.transform(b)
        return a, b, label

class SingleImageDataset(Dataset):
    """For embed_all: iterate through image list"""
    def __init__(self, image_paths, transform=None):
        self.paths = image_paths
        self.transform = transform if transform is not None else get_transforms(train=False)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        return self.transform(img), p
