from torchvision import transforms

from autobias.utils.configured import Configured

RESNET_NORMALIZE = transforms.Normalize(
  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)


class Cifar10AugmentedTransform(Configured):
  def __init__(self):
    self.transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, 4)
    ])

  def __call__(self, x):
    return self.transform(x)


class ImageNetEvalTransform(Configured):
  def __init__(self, center_to_crop=224, resize=True):
    self.resize = resize
    self.center_to_crop = center_to_crop
    all_transforms = []
    if resize:
      all_transforms.append(transforms.Resize(256))
    all_transforms += [
      transforms.CenterCrop(center_to_crop),
      transforms.ToTensor(),
      RESNET_NORMALIZE
    ]
    self.transform = transforms.Compose(all_transforms)

  def __call__(self, x):
    return self.transform(x)


class RandomCropTransform(Configured):
  def __init__(self, crop_size, horizontal_flip=True):
    self.crop_size = crop_size
    self.horizontal_flip = horizontal_flip
    to_compose = []
    if self.crop_size is not None:
      to_compose.append(transforms.RandomCrop(self.crop_size))
    if self.horizontal_flip:
      to_compose.append(transforms.RandomHorizontalFlip())

    to_compose += [
      transforms.ToTensor(),
      RESNET_NORMALIZE
    ]

    self.transform = transforms.Compose(to_compose)

  def __call__(self, x):
    return self.transform(x)



