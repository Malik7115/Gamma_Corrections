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


model       = gammaModel1()
criterion   = nn.MSELoss()
optimizer   = optim.Adam(model.parameters(), lr= 0.005)
threshold   = 0.02
epochs      = 100
batch_size  = 25
dummy_input = example = torch.rand(1, 1, 100, 100)


csv_path        = "/home/ibrahim/Projects/Datasets/HPO_Recording/Images/Annotations.csv"
images_path     = "/home/ibrahim/Projects/Datasets/HPO_Recording/Images/"
model_save_path = "/home/ibrahim/Projects/Gamma_Corrections/Pytorch/SavedModels/"

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


best_acc       = 0.0
best_acc_train = 0.0
lowest_loss    = 100
acc_threshold  = 0.15

for epoch in range(epochs):
    running_loss_train = 0.0
    running_loss_test  = 0.0

    running_accuracy_train = 0.0
    running_accuracy_test  = 0.0

    total                  = 0.0

    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        
        inputs, targets = batch
        preds           = model(inputs)
        
        optimizer.zero_grad()
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss_train         += loss.item() * inputs.size(0)
        running_accuracy_train     += (abs(preds-targets) <= threshold).sum().item()

    model.eval()
    for i_batch, batch in enumerate(test_dataloader):
        
        with torch.no_grad():
            inputs, targets = batch
            preds           = model(inputs)

            running_loss_test         += loss.item() * inputs.size(0)
            running_accuracy_test     += (abs(preds-targets) <= threshold).sum().item()
        

    epoch_accuracy_train = running_accuracy_train/len(train_dataloader.dataset)
    epoch_loss_train     = running_loss_train/len(train_dataloader.dataset)

    epoch_accuracy_test  = running_accuracy_test/len(test_dataloader.dataset)

    # if (epoch_accuracy_test > best_acc):
    #     best_acc = epoch_accuracy_test

    if(epoch_accuracy_train >= 0.90 and \
        abs(epoch_accuracy_train - epoch_accuracy_test) <= acc_threshold):

        acc_threshold = abs(epoch_accuracy_train - epoch_accuracy_test)
        print("here" + "\t" + str(acc_threshold) + "\t" + str(epoch))
        best_acc = epoch_accuracy_test

        torch.onnx.export(
                            model,       
                            inputs,                         
                            model_save_path + "gamma_correction.onnx",   
                            export_params=True,        
                            opset_version=10,          
                            do_constant_folding=True,  
                            input_names = ['input'],   
                            output_names = ['output'], 
                            dynamic_axes={'input' : {0 : 'batch_size'},
                            'output' : {0 : 'batch_size'}}
                  )
        

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_save_path + "gamma_correction.pth")

        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(model_save_path + "gamma_model_cpp.pt")


    # print("epoch loss " + str(epoch) + "  " + str(epoch_loss_train))
    print(str(epoch) + "\t" + str(best_acc)\
        + "\t" + str(epoch_accuracy_test) + "\t" +\
        str(epoch_accuracy_train))


    # print("epoch_accuracy_train:  " + str(epoch_accuracy_train)\
    #      + "\t" + "best:  " + str(best_acc))



