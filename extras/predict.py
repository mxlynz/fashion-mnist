import glob
import torch
import os

from data import validation_loader, test_loader, classes
from model import MODELS

loader = validation_loader  # switch to test_loader only for the final evaluation

RUN = None  # e.g. 'v2/02_bs64_adam_lr0.0005_20261112_103200'; None = newest
path = f'models/{RUN}/best.pt' if RUN else max(glob.glob('models/**/best.pt', recursive=True), key=os.path.getmtime)
ckpt = torch.load(path)
model = MODELS[ckpt['arch']]()
model.load_state_dict(ckpt['state_dict'])
model.eval()

images, labels = next(iter(loader))
with torch.no_grad():
    outputs = model(images)
predicted = outputs.argmax(dim=1)  # index of the highest logit = predicted class

print(f'checkpoint: {path} (epoch {ckpt["epoch"]})')
for p, l in list(zip(predicted, labels))[:8]:
    print(f'predicted: {classes[p]:12} actual: {classes[l]}')

confusion = torch.zeros(10, 10, dtype=torch.int64)  # rows: actual, columns: predicted
correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader:                        # all images of the chosen set
        predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum().item()    # count matches in this batch
        total += labels.size(0)
        for l, p in zip(labels, predicted):              # conf mat
            confusion[l, p] += 1
print(f'accuracy: {correct / total:.2%} ({correct}/{total})')
print('\nconfusion matrix (rows: actual, columns: predicted)')
print(' ' * 14 + ''.join(f'{i:>6}' for i in range(10)))
for i, row in enumerate(confusion):
    print(f'{i} {classes[i]:12}' + ''.join(f'{n:>6}' for n in row.tolist()))