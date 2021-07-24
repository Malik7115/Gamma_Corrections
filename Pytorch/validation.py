import enum
import os
import torch
import torch.onnx
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader , random_split
from torchvision import transforms, utils
from models import *
from cDataLoaders import*
os.system('clear')

model_save_path =\
     "/home/ibrahim/Projects/Gamma_Corrections/Pytorch/SavedModels/gamma_correction.pth"
checkpoint = torch.load(model_save_path)


model = gammaModel1()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

input_transforms = transform= transforms.Compose([RescaleCrop(100)\
                                                ,toTensor()])

lookUpTable = np.empty((1,256), np.uint8)


cv2.namedWindow("corrupted")
cv2.namedWindow("processed")


img_names = os.listdir(images_path)

for i in range(10):
    
     img = cv2.imread(images_path + img_names[i], 0)
     print(img.shape)
     model_in = input_transforms(img)
     model_in = torch.unsqueeze(model_in,0)

     pred = model(model_in)
     pred = pred.detach().numpy()
     print(pred)

     for i in range(256):
          lookUpTable[0,i] = np.clip(pow(i / 255.0, 1/pred) * 255.0, 0, 255)

     procFrame = cv2.LUT(img, lookUpTable)

     cv2.imshow("corrupted", img)
     cv2.imshow("processed", procFrame)
     cv2.waitKey(0)

cv2.destroyAllWindows()