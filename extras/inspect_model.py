import glob
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import validation_set, classes
from model import MODELS

RUN = None  # e.g. 'v2/02_bs64_adam_norm_ep30_20261113_175222'; None = newest
CLASS = 6   # 0 T-shirt, 2 Pullover, 4 Coat, 6 Shirt, ...

path = f'models/{RUN}/best.pt' if RUN else max(glob.glob('models/**/best.pt', recursive=True), key=os.path.getmtime)
ckpt = torch.load(path)
model = MODELS[ckpt['arch']]()
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f'checkpoint: {path} (epoch {ckpt["epoch"]})')


def show(t, title, **kw):  # t: C×H×W, one subplot per channel
    c = t.shape[0]
    cols = min(c, 8)
    rows = (c + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k < c:
            ax.imshow(t[k], **kw)
            ax.set_title(str(k), fontsize=8)
        ax.axis('off')
    fig.suptitle(title)


# 1) learned conv1 filters: red = positive weight, blue = negative weight
w = model.conv1.weight.detach()[:, 0]  # C×3×3
lim = w.abs().max()
show(w, 'conv1 filters', cmap='bwr', vmin=-lim, vmax=lim)

# 2) feature maps after BN + ReLU for the first image of the chosen class
idx = next(i for i in range(len(validation_set)) if validation_set[i][1] == CLASS)
img, label = validation_set[idx]

maps = {}
for name in ['bn1', 'bn2', 'bn3']:  # v2 layer names
    getattr(model, name).register_forward_hook(
        lambda m, inp, out, name=name: maps.__setitem__(name, F.relu(out)[0]))
with torch.no_grad():
    model(img.unsqueeze(0))

plt.figure(figsize=(2, 2))
plt.imshow(img[0], cmap='gray')
plt.title(classes[label])
plt.axis('off')
for name, m in maps.items():
    show(m, f'{name} + ReLU: {tuple(m.shape)}')

plt.show()