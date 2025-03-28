import torch
import torch.nn as nn


class LinearInterpolationModule(nn.Module):
    def __init__(self, y_points, device):
        super(LinearInterpolationModule, self).__init__()
        self.device = device
        self.y_points = y_points.to(device)

    def forward(self, x_new_):
        x_new = x_new_.to(self.device)
        batch_size, num_points = self.y_points.shape
        x_points = torch.linspace(0, 1, num_points).to(self.device).expand(batch_size, -1).contiguous()
        x_new_expanded = x_new.unsqueeze(0).expand(batch_size, -1).contiguous()
        idxs = torch.searchsorted(x_points, x_new_expanded, right=True)
        idxs = idxs - 1
        idxs = idxs.clamp(min=0, max=num_points - 2)
        x1 = torch.gather(x_points, 1, idxs)
        x2 = torch.gather(x_points, 1, idxs + 1)
        y1 = torch.gather(self.y_points, 1, idxs)
        y2 = torch.gather(self.y_points, 1, idxs + 1)
        weights = (x_new_expanded - x1) / (x2 - x1)
        y_interpolated = y1 + weights * (y2 - y1)
        return y_interpolated


class IndexModule(nn.Module):
    def __init__(self, y_points, device):
        super(IndexModule, self).__init__()
        self.device = device
        self.linterp = LinearInterpolationModule(y_points, self.device)
        self.delta = (1/4200)*5
        self.slots = 4
        self.offsets = torch.linspace(-self.delta, self.delta, self.slots).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        indices = x.shape[0]
        x = x.view(-1, 1)
        x = (x + self.offsets).flatten()
        z = x.reshape(-1, self.slots)
        for i in range(len(z)):
            for j in range(len(z[i])):
                print(f'{i} {j}: {z[i][j]}')
            print("\n\n")
        x = self.linterp(x)
        x = x.reshape(indices, -1)
        return x


y_points = torch.tensor([[0.0, 1.0, 0.0]])
x_new = torch.tensor([0,0.25,0.5,0.75,1])

module = IndexModule(y_points, device='cuda:0')
output = module(x_new)

for i in range(len(output)):
    for j in range(len(output[i])):
        print(f'{i} {j}: {output[i][j]}')
    print("\n\n")