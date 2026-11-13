import matplotlib.pyplot as plt
import numpy as np
import torchvision

from data import training_loader, classes

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img * 0.3528 + 0.2856
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(training_loader) # fetch only one batch
images, labels = next(dataiter)

# Create a grid from the images and show them
print(images.shape, images.min().item(), images.max().item())
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print('  '.join(classes[labels[j]] for j in range(4)))
plt.show()
