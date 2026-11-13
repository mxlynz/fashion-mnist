import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) #  keeps max of each 2×2 window, halves height & width
        self.conv2 = nn.Conv2d(6, 16, 5) # 16x8x8 
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # combination of 256 features into 120...
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 logits, one per class (no ReLU)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # 16×4×4
        x = x.view(-1, 16 * 4 * 4) # flatten into 1d vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OurClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)

        x = x.squeeze([2, 3])
        x = self.fc(x)
        return x



MODELS = {
    'lenet': GarmentClassifier,
    'v2': OurClassifier,
}
# Output size of a conv/pool layer (per side):
#   out = (in - kernel + 2*padding) / stride + 1
# in - kernel: how far the kernel can move before it sticks out past the edge
# / stride:    how many steps fit in that distance
# + 1:         counts the starting position
# conv2: (12 - 5 + 0) / 1 + 1 = 8   -> 16×8×8
# pool:  ( 8 - 2 + 0) / 2 + 1 = 4   -> 16×4×4
# Channels = number of filters
# height / widht with in - kernel + 1 (with no padding and stride 1)