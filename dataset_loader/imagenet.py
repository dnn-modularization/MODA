from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from dataset_loader.dataset_utils import create_dataset_loader

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

def _get_transforms():  
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf

class ImageNetFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        self.imgs = self.samples
    
    @property
    def data(self):
        return np.array([path for path, _ in self.samples])
    
    @data.setter
    def data(self, new_data):
        if not hasattr(self, '_path_to_label'):
            self._path_to_label = {path: label for path, label in self.samples}
        self.samples = [(path, self._path_to_label[path]) for path in new_data]
        self.imgs = self.samples
    
    @property
    def targets(self):
        return np.array([label for _, label in self.samples])
    
    @targets.setter
    def targets(self, new_targets):
        assert len(new_targets) == len(self.samples), "New targets must match the number of samples."
        self.samples = [(path, target) for (path, _), target in zip(self.samples, new_targets)]
        self.imgs = self.samples


def load_imagenet_dataset(batch_size, dataset_dir, train_augmentation=True, 
                          num_workers=2, target_classes=None, sample_size_per_class=None):
    train_tf, val_tf = _get_transforms()
    if not train_augmentation:
        print("Training data transform/augmentation is disabled. Using validation augmentation instead.")
        train_tf = val_tf
    train_ds = ImageNetFolder(root=str(Path(dataset_dir) / 'train'), transform=train_tf)
    val_ds = ImageNetFolder(root=str(Path(dataset_dir) / 'val'), transform=val_tf)
    return create_dataset_loader(train_dataset=train_ds, test_dataset=val_ds, 
                                 sample_size_per_class=sample_size_per_class, batch_size=batch_size, 
                                 num_workers=num_workers, target_classes=target_classes)
