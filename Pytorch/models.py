from torch import nn
import numpy as np
import torch
import torch.nn.functional as F



class gammaModel1(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100*100, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)

        x = F.relu(self.fc2(x))
        # x = self.dropout(x)

        x = F.relu(self.fc3(x))
        # x = self.dropout(x)

        x = F.relu(self.fc4(x))
        # x = self.dropout(x)

        x = F.relu(self.fc5(x))
        # x = self.dropout(x)

        x = self.fc6(x)
        x = torch.squeeze(x)
        return x





class gammaModel2(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100*100, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        


    def forward(self, x):

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        # x = self.dropout(x)

        x = F.relu(self.fc2(x))
        # x = self.dropout(x)

        x = F.relu(self.fc3(x))
        # x = self.dropout(x)

        x = F.relu(self.fc4(x))
        # x = self.dropout(x)

        x = F.relu(self.fc5(x))
        # x = self.dropout(x)

        x = torch.squeeze(x)
        return x


