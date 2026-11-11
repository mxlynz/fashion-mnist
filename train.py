import torch
import os

from data import training_loader, validation_loader
from model import GarmentClassifier
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.manual_seed(0) # zum testen bzw vergleichen

model = GarmentClassifier()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    epoch_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data

        # Zero your gradients for every batch!
        # to replace instead of adding gradients 
        optimizer.zero_grad()

        outputs = model(inputs) # forward pass (liefert logits)

        loss = loss_fn(outputs, labels) # fehler 
        loss.backward() # computation of gradient of loss / backpropagation

        # Adjust learning weights
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return epoch_loss / len(training_loader)  # average over whole epoch

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch_number + 1}:')

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss} valid {avg_vloss}')

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        os.makedirs('models', exist_ok=True)                     # legt models/ an, falls nicht vorhanden
        model_path = f'models/model_{timestamp}_{epoch_number}'  # speichert in models/
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

