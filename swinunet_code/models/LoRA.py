import torch
from torch import nn
import numpy as np

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r, alpha):
        super(LoRALinear, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha

        self.lora_a = nn.Parameter(torch.zeros((original_layer.out_features,r)))
        self.lora_b = nn.Parameter(torch.zeros((r, original_layer.in_features)))

        nn.init.kaiming_uniform_(self.lora_a, a = np.sqrt(5))
        nn.init.zeros_(self.lora_b)

        self.scaling = self.alpha / self.r

    def forward(self, x):
        lora_a = self.lora_a.to(x.device)
        lora_b = self.lora_b.to(x.device)

        output = self.original_layer(x)

        if x.dim() == 3:
            lora_output = (x @ lora_b.t() @ lora_a.t()) * self.scaling
        else:
            lora_output = (x @ lora_b.T @ lora_a.T) * self.scaling
        
        return output + lora_output

  

