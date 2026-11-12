import torch, torchvision
from torchvision.transforms import v2
from data import training_set

raw = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
x = torch.stack([raw[i][0] for i in training_set.indices])  # only the 54,000 training images
print(f'mean: {x.mean().item():.4f}  std: {x.std().item():.4f}')