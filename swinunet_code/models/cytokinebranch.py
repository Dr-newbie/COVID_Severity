import pandas as pd
import torch
import numpy as np
from torch import nn


class CytokineBranch(nn.Module):
    def __init__(self, input_size, num_classes=3):
        super(CytokineBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.InstanceNorm1d(128),
            nn.Dropout(0.3), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.InstanceNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),  
            nn.ReLU(),
            nn.InstanceNorm1d(32),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        features = self.fc(x)
        output = self.classifier(features)
        return output