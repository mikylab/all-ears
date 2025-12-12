import d2l
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch import nn
import random
from torchvision import transforms


class EarData(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(128, 128), train_split_size=0.7):
        super().__init__()
        self.save_hyperparameters()

        transform = d2l.transforms.Compose([d2l.transforms.Resize(resize), d2l.transforms.ToTensor()])
        data = []
        self.names = []

        images_folder = "EarVN1.0/Images"
        for ear_folder_name in sorted(os.listdir(images_folder)):
            parts = ear_folder_name.split(".")
            label, name = parts
            label = int(label) - 1
            self.names.append(name)

            ear_folder = images_folder + "/" + ear_folder_name
            for image_name in os.listdir(ear_folder):
                with Image.open(ear_folder + "/" + image_name) as image:
                    transformed_image = transform(image)
                    data.append((transformed_image, label))

        split = int(len(data) * train_split_size)
        self.train = data[:split]
        self.val = data[split:]

    def text_labels(self, indices):
        return [self.names[i] for i in indices]

    def get_dataloader(self, train=True):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train)

    def visualize(self, batch, nrows=2, ncols=2):
        X, y = batch
        X = X.permute(0, 2, 3, 1)
        labels = self.text_labels(y)
        d2l.show_images(X, nrows, ncols, titles=labels)


class PairedEarData(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(128, 128), train_split_size=0.7):
        super().__init__()
        self.save_hyperparameters()

        random.seed(42)

        transform = d2l.transforms.Compose([d2l.transforms.Resize(resize), d2l.transforms.ToTensor()])

        images = []
        last_person_index = -1

        images_folder = "EarVN1.0/Images"
        ear_folders = os.listdir(images_folder)
        # shuffling so that doing a split off person index doesn't only split off one gender
        random.shuffle(ear_folders)

        for index, ear_folder_name in enumerate(ear_folders):
            ear_folder = images_folder + "/" + ear_folder_name
            for image_name in os.listdir(ear_folder):
                with Image.open(ear_folder + "/" + image_name) as image:
                    transformed_image = transform(image)
                    images.append((transformed_image, index))

        # splitting by person to prevent leakage between sets
        person_split = int(train_split_size * len(ear_folders))
        train_positive = []
        train_negative = []
        val_positive = []
        val_negative = []

        for i in range(len(images)):
            image1, person_index1 = images[i]

            for j in range(i + 1, len(images)):
                image2, person_index2 = images[j]
                image_pair = image1, image2

                label = int(person_index1 == person_index2)
                train = person_index1 < person_split and person_index2 < person_split
                val = person_index1 >= person_split and person_index2 >= person_split

                if label == 1:
                    if random.random() > 0.03:
                        continue
                else:
                    # there will significantly more negative examples without sampling less
                    if random.random() > 0.0004:
                        continue

                if train:
                    if label == 1:
                        train_positive.append((image_pair, label))
                    else:
                        train_negative.append((image_pair, label))
                elif val:
                    if label == 1:
                        val_positive.append((image_pair, label))
                    else:
                        val_negative.append((image_pair, label))
                else:
                    # one of the images would leak into the wrong set if
                    # we go down this branch
                    continue

        print(f"train_positive={len(train_positive)}")
        print(f"train_negative={len(train_negative)}")
        print(f"val_positive={len(val_positive)}")
        print(f"val_negative={len(val_negative)}")

        # truncating to not imbalance classes, can remove this
        length = min(len(train_positive), len(train_negative))
        self.train = train_positive[:length] + train_negative[:length]
        length = min(len(val_positive), len(val_negative))
        self.val = val_positive[:length] + val_negative[:length]

        print(f"train={len(self.train)}")
        print(f"val={len(self.val)}")

    def get_dataloader(self, train=True):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train)

    def get_label(self, class_value):
        return "positive" if class_value == 1 else "negative"

    def visualize(self, batch):
        (X_1, X_2), y = batch
        X_1 = X_1[0]
        X_1 = X_1.permute(1, 2, 0)
        X_2 = X_2[0]
        X_2 = X_2.permute(1, 2, 0)
        y = y[0]
        label = self.get_label(y)
        d2l.show_images([X_1, X_2], 1, 2, titles=[label, None])

class GenderedEarData(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(128, 128), train_split_size=0.7):
        super().__init__()
        self.save_hyperparameters()

        
        random_transforms = [transforms.ColorJitter(brightness=.5, hue=.2), 
                            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
                            ]
        transform= d2l.transforms.Compose([d2l.transforms.Resize(resize),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomApply(transforms=random_transforms, p=0.5), 
                                           transforms.RandomPerspective(distortion_scale=0.4, p=.25),
                                            d2l.transforms.ToTensor()])
        transform_valid = d2l.transforms.Compose([d2l.transforms.Resize(resize),
                                            d2l.transforms.ToTensor()])

        data = []
        val_data = []
        self.names = []

        images_folder = "EarVN1.0/Images"
        for ear_folder_name in sorted(os.listdir(images_folder)):
            parts = ear_folder_name.split(".")
            label, name = parts
            label = 0 if int(label) >= 98 else 1
            self.names.append(name)

            ear_folder = images_folder + "/" + ear_folder_name
            for i, image_name in enumerate(os.listdir(ear_folder)):
                with Image.open(ear_folder + "/" + image_name) as image:
                    imageSet= len(os.listdir(ear_folder)) * self.train_split_size
                    if i < imageSet:
                        transformed_image = transform(image)
                        data.append((transformed_image, label))
                        transformed_image = transform_valid(image)
                        data.append((transformed_image, label))
                    else:
                        transformed_image = transform_valid(image)
                        val_data.append((transformed_image, label))

        self.train = data
        self.val = val_data
    
    def text_labels(self, indices):
        return [self.names[i] for i in indices]

    def get_dataloader(self, train=True):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train)

    def visualize(self, batch, nrows=2, ncols=2):
        X, y = batch
        X = X.permute(0, 2, 3, 1)
        labels = self.text_labels(y)
        d2l.show_images(X, nrows, ncols, titles=labels)
