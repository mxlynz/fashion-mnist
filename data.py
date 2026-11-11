import torch
import torchvision
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])
# .ToDtype = pixel values as 0.0 to 1.0, through .Normalize from range -1 to 1 -> für ausgewogeneren Gradienten

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
# donwload=false -> 10,000 test image (validation), =true -> 60,000 training images

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

if __name__ == '__main__':
    print(f'Training set has {len(training_set)} instances')
    print(f'Validation set has {len(validation_set)} instances')


# improvement
# adjust batch_size; depends on processor architecture, available memory and effect on training convergency