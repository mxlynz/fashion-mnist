import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 6×24×24 number feature maps x height x width
        self.pool = nn.MaxPool2d(2, 2) #  keeps max of each 2×2 window, halves height & width
        self.conv2 = nn.Conv2d(6, 16, 5) # 16x8x8 
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # combination of 256 features into 120...
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 logits, one per class (no ReLU)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 6×12×12
        x = self.pool(F.relu(self.conv2(x))) # 16×4×4
        x = x.view(-1, 16 * 4 * 4) # flatten into 1d vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Output size of a conv/pool layer (per side):
#   out = (in - kernel + 2*padding) / stride + 1
# in - kernel: how far the kernel can move before it sticks out past the edge
# / stride:    how many steps fit in that distance
# + 1:         counts the starting position
# conv2: (12 - 5 + 0) / 1 + 1 = 8   -> 16×8×8
# pool:  ( 8 - 2 + 0) / 2 + 1 = 4   -> 16×4×4
# Channels = number of filters
# height / widht with in - kernel + 1 (with no padding and stride 1)