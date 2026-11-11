import torch
import torchvision
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])
# .ToDtype = pixel values as 0.0 to 1.0, through .Normalize from range -1 to 1 -> für ausgewogeneren Gradienten

full_training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
test_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
# train=True -> 60,000 training images, train=False -> 10,000 test images
# download=True only downloads if not already present

# Split training images into 54,000 train / 6,000 validation; fixed seed -> same split every run
training_set, validation_set = torch.utils.data.random_split(
    full_training_set, [54_000, 6_000], generator=torch.Generator().manual_seed(42))

training_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)  # only for final evaluation

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

if __name__ == '__main__':
    print(f'Training set has {len(training_set)} instances')
    print(f'Validation set has {len(validation_set)} instances')


# improvement
# adjust batch_size; depends on processor architecture, available memory and effect on training convergency