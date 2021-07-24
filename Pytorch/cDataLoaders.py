import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import cv2

os.system('clear')

csv_path        = "/home/ibrahim/Projects/Datasets/HPO_Recording/Images/Annotations.csv"
images_path     = "/home/ibrahim/Projects/Datasets/HPO_Recording/Images/"
model_save_path = "/home/ibrahim/Projects/Gamma_Corrections/Pytorch/SavedModels/"
batch_size = 25


class gammaCorrectionDataset(Dataset):
    def __init__(self, csv_file, imgs_path, transform = None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.imgs_path   = imgs_path
        self.transform   = transform
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.imgs_path,
                                self.annotations.iloc[idx, 0]) 
        img      = io.imread(img_name)
        gamma    = self.annotations.iloc[idx, 1]
        gamma    = torch.tensor(gamma, dtype= torch.float32)

        if self.transform:
            img = self.transform(img)
        return img, gamma





class RescaleCrop():
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        img = img[20:270, 40:320]
        img = cv2.resize(img, (self.output_size, self.output_size))
        img = np.expand_dims(img, axis= 2)

        return img

# class Crop():
#     def __call__(self, img):
#         return img.roi(cv2.rect(40,20,320,270))

class toTensor():
    def __call__(self, img):
        img = img.transpose((2,0,1))
        img = img/255.0
        return torch.tensor(img, dtype= torch.float32)


dataset   = gammaCorrectionDataset(csv_path, images_path,\
                transform= transforms.Compose([RescaleCrop(100)\
                                                ,toTensor()]))
train, test     = random_split(dataset, [int(0.8*len(dataset)),\
                  (int(len(dataset)) - int(0.8*len(dataset)))],\
                  generator=torch.Generator().manual_seed(42))


train_dataloader   = DataLoader(train, batch_size,\
                     shuffle= True, num_workers= 6)                        
test_dataloader    = DataLoader(test,  batch_size,\
                     shuffle= True, num_workers= 6)

