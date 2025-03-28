import torch
import torch.nn as nn


cnn = nn.Sequential(
    nn.Conv1d(1, 8, kernel_size=8, stride=1, padding=0),
    nn.LeakyReLU(),
    nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
    nn.Flatten(start_dim=1),
    nn.LayerNorm(24)
)


x = torch.randn(10, 1, 32)

y = cnn(x)
print(y.shape)