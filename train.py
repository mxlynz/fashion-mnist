import torch
import os
import time

from data import training_loader, validation_loader
from model import GarmentClassifier
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

RUN_NAME = '03_bs64_adam_norm'
LOG_EVERY = 100  # batches between Loss/train logs (843 batches per epoch now)

torch.manual_seed(0) # zum testen bzw vergleichen

model = GarmentClassifier()
loss_fn = torch.nn.CrossEntropyLoss() # softmax(logits->Wslk.)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    epoch_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data

        # Zero your gradients for every batch!
        # to replace instead of adding gradients 
        optimizer.zero_grad()

        outputs = model(inputs) # forward pass (liefert logits für Vorhersage)

        loss = loss_fn(outputs, labels) # fehler 
        loss.backward() # computation of gradient of loss / backpropagation

        # Adjust learning weights -> Gewichte in Richtung -Gradient
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % LOG_EVERY == LOG_EVERY - 1:
            last_loss = running_loss / LOG_EVERY # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return epoch_loss / len(training_loader)  # average over whole epoch

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/v1/{RUN_NAME}_{timestamp}')
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

start = time.time()

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch_number + 1}:')

    # Make sure gradient tracking is on, and do a pass over the data -> 
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


print(f'training time: {(time.time() - start) / 60:.1f} min')

