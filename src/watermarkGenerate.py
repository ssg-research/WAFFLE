import torch
import torchvision as tv
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import data_handle as dh
from Network import CIFAR10_BASE, SimpleDataset
import logger
import random
import os
import numpy as np
from PIL import Image

log = logger.Logger(prefix=">>>")

transform = transforms.Compose(
    [
     tv.transforms.Resize(256),
     tv.transforms.CenterCrop(256),
     transforms.ToTensor(),
     transforms.Normalize((1, 1, 1), (1, 1, 1))])

watermarkset = dh.ImageNet('./data/IMAGENET/', train=False,
                                       download=True, transform = transform)

def construct_watermark_set(watermark_set: data.Dataset, watermark_size: int, number_of_classes: int,
                            partition: bool) -> (data.Dataset, data.Dataset):
    len_ = watermark_set.__len__()
    watermark, train = torch.utils.data.dataset.random_split(watermark_set, (watermark_size, len_ - watermark_size))
    log.info("Split set into: {} and {}".format(len(watermark), len(train)))
    all_label = another_label(watermark_size, number_of_classes)
    for index, ((img, label), i) in enumerate(zip(watermark, all_label)):
        img = np.array(img)

        img = np.transpose(img * 255, (1, 2, 0))
        img = ((img - img.min()) * (1 / (img.max() - img.min())) * 255)
        # print(data)
        image = Image.fromarray(img.astype(dtype='uint8'), 'RGB')
        # image.save("./data/datasets/CIFAR&PATTERN/" + str(idx) + "/wm_" + str(i + 1) + ".png")
        newpath = './data/datasets/IMAGENET/' + str(i)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        image.save('./data/datasets/IMAGENET/' + str(i) + '/' + str(index) + '.png')
    # watermark = SimpleDataset([(img, i) for (img, label), i in zip(watermark, all_label)])
    # if partition:
    #     return watermark, train
    # else:
    #     return watermark, None

def another_label(data_size: int, number_of_classes: int) :
    data_per_class = int(data_size / number_of_classes)
    all_label = []
    for i in range(number_of_classes):
        for k in range(data_per_class):
            all_label.append(i)

    return all_label


# def create_random_image(image_size, watermark_size):
#     for i in range(watermark_size):
#       images = torch.randn(image_size.height, image_size.length)

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class FakeData(VisionDataset):

    """A fake dataset that returns randomly generated images and returns them as PIL images
    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the datset. Default: 10
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0
    """

    def __init__(self, size=100, image_size=(3, 32, 32), num_classes=10,
                 transform=None, target_transform=None, random_offset=0):
        super(FakeData, self).__init__(None, transform=transform,
                                       target_transform=target_transform)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)

        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.size




class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

def generated_random_watermark(size:int, image_size:tuple, num_classes:int):
    for i in range(num_classes):
        newpath = './data/datasets/RANDOM/'+str(i)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    random_image = FakeData(size=size, image_size=image_size, num_classes=num_classes)
    for index, (image, target) in enumerate(random_image, 0):
       image.save('./data/datasets/RANDOM/'+str(target.item())+'/'+str(index)+'.png')

# def generated_imagenet_watermark(size:int, num_classes:int):



# generated_random_watermark(size= 200, image_size=(3,32,32), num_classes=10)

construct_watermark_set(watermarkset, 100, 10, partition = False)


# torch.save(watermarkset, "./data/datasets/imagenet")
# torch.save(FakeData, "random_image")