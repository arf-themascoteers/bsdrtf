import torch
import torch.nn as nn


cnn = nn.Sequential(
    nn.Conv1d(1, 24, kernel_size=32, stride=32, padding=0),
    nn.LeakyReLU(),
    nn.Flatten(start_dim=1),
    nn.LayerNorm(24)
)


x = torch.randn(10, 1, 32)

y = cnn(x)
print(y.shape)