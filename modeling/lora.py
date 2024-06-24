import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim,rank, alpha=0.01):
        super(LoRALayer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank)* std_dev)
        self.B = nn.Parameter(torch.zeros(rank,out_dim))
        self.alpha = alpha
    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

class LinearWithLoRA(nn.Module):
    def __init__(self, linear,rank, alpha=0.01):
        super(LinearWithLoRA, self).__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False
            
        self.lora = LoRALayer(linear.in_features, linear.out_features,rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)
