import glob
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import validation_set, classes
from model import GarmentClassifier

path = max(glob.glob('models/model_*'), key=os.path.getmtime)  # newest checkpoint
model = GarmentClassifier()
model.load_state_dict(torch.load(path))
model.eval()

# 1) learned conv1 filters: red = positive weight, blue = negative weight
w = model.conv1.weight.detach()[:, 0]  # 6×5×5
lim = w.abs().max()
fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
for k, ax in enumerate(axes):
    ax.imshow(w[k], cmap='bwr', vmin=-lim, vmax=lim)
    ax.set_title(f'filter {k}')
    ax.axis('off')
fig.suptitle('conv1 filters')

# 2) feature maps for one example image (change index to look at other images)
img, label = validation_set[2]  # 1×28×28 # 0 statt x
with torch.no_grad():
    maps1 = F.relu(model.conv1(img.unsqueeze(0)))       # 1×6×24×24
    maps2 = F.relu(model.conv2(model.pool(maps1)))      # 1×16×8×8

fig, axes = plt.subplots(1, 7, figsize=(14, 2.5))
axes[0].imshow(img[0], cmap='gray')
axes[0].set_title(classes[label])
for k in range(6):
    axes[k + 1].imshow(maps1[0, k])
    axes[k + 1].set_title(f'map {k}')
for ax in axes:
    ax.axis('off')
fig.suptitle('conv1 feature maps (after ReLU)')

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for k, ax in enumerate(axes.flat):
    ax.imshow(maps2[0, k])
    ax.set_title(f'map {k}')
    ax.axis('off')
fig.suptitle('conv2 feature maps (after ReLU)')

plt.show()