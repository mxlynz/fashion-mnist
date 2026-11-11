import glob
import torch
import os

from data import validation_loader, classes
from model import GarmentClassifier

path = max(glob.glob('models/model_*'), key=os.path.getmtime) # newest checkpoint = best one
model = GarmentClassifier()
model.load_state_dict(torch.load(path))
model.eval()

images, labels = next(iter(validation_loader))  # first batch of 4
with torch.no_grad():
    outputs = model(images)
predicted = outputs.argmax(dim=1)  # index of the highest logit = predicted class

print(f'checkpoint: {path}')
for p, l in zip(predicted, labels):
    print(f'predicted: {classes[p]:12} actual: {classes[l]}')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in validation_loader:            # all 10,000 validation images
        predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum().item()   # count matches in this batch
        total += labels.size(0)                          # batch size (4)
print(f'accuracy: {correct / total:.2%} ({correct}/{total})')