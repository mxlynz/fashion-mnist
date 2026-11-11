import glob
import torch
import os

from data import validation_loader, test_loader, classes
from model import GarmentClassifier

loader = validation_loader  # switch to test_loader only for the final evaluation

path = max(glob.glob('models/model_*'), key=os.path.getmtime)  # newest checkpoint = best of latest run
model = GarmentClassifier()
model.load_state_dict(torch.load(path))
model.eval()

images, labels = next(iter(loader))  # first batch of 4
with torch.no_grad():
    outputs = model(images)
predicted = outputs.argmax(dim=1)  # index of the highest logit = predicted class

print(f'checkpoint: {path}')
for p, l in zip(predicted, labels):
    print(f'predicted: {classes[p]:12} actual: {classes[l]}')

confusion = torch.zeros(10, 10, dtype=torch.int64)  # rows: actual, columns: predicted
correct = 0
total = 0
with torch.no_grad():
    for images, labels in loader:                        # all images of the chosen set
        predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum().item()    # count matches in this batch
        total += labels.size(0)                          # batch size (4)
        for l, p in zip(labels, predicted):              # conf mat
            confusion[l, p] += 1
print(f'accuracy: {correct / total:.2%} ({correct}/{total})')
print('\nconfusion matrix (rows: actual, columns: predicted)')
print(' ' * 14 + ''.join(f'{i:>6}' for i in range(10)))
for i, row in enumerate(confusion):
    print(f'{i} {classes[i]:12}' + ''.join(f'{n:>6}' for n in row.tolist()))