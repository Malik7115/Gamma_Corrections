import enum
import os
import torch
import torch.onnx
import onnx
import onnxruntime
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
     "/home/ibrahim/Projects/Gamma_Corrections/Pytorch/SavedModels/"

pt_model_name   = "gamma_correction.pth"
onnx_model_name = "gamma_correction.onnx"

checkpoint = torch.load(model_save_path + pt_model_name)

onnx_model = onnx.load(model_save_path + onnx_model_name)
ort_session = onnxruntime.InferenceSession(model_save_path + onnx_model_name)
onnx.checker.check_model(onnx_model)

model = gammaModel1()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

input_transforms = transform= transforms.Compose([RescaleCrop(100)\
                                                ,toTensor()])

lookUpTable = np.empty((1,256), np.uint8)

img_names = os.listdir(images_path)


cv2.namedWindow("corrupted")
cv2.namedWindow("processed")

input_name = ort_session.get_inputs()[0].name
label_name = ort_session.get_outputs()[0].name

x = torch.randn(25, 1, 100, 100, requires_grad=True)

for i in range(1):
    
     img = cv2.imread(images_path + img_names[i], 0)
     print(img.shape)
     model_in = input_transforms(img)
     model_in = torch.unsqueeze(model_in,0)

     # ort_inputs = {ort_session.get_inputs()[0].name: x.detach().numpy()}
     ort_inputs = {ort_session.get_inputs()[0].name: model_in.detach().numpy()}
     ort_outs = ort_session.run(None, ort_inputs)

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

print(ort_outs)

print(onnxruntime.__version__)
