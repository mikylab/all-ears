import d2l
import torch
from torch import nn
import torchvision


class CNNbaseline(d2l.Classifier):
    """The LeNet-5 model."""
    def __init__(self, num_classes=164):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(84), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))


# This source showed a way to set up these classifiers for a simple siamese network
# I mean simple since it uses BCELoss instead of making triplets
# https://github.com/pytorch/examples/blob/main/siamese_network/README.md

class PairCNN(d2l.Classifier):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(32, kernel_size=5, padding=2), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.LazyLinear(120), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(84), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_1, X_2):
        output_1 = self.cnn(X_1)
        output_2 = self.cnn(X_2)
        output = torch.cat((self.flatten(output_1), self.flatten(output_2)), 1)
        output = self.fc(output)
        return self.sigmoid(output)


class PairResnet18(d2l.Classifier):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        fc_in_features = self.resnet.fc.in_features
        # remove the last layer so that we can combine the output from two images
        # before passing through the fully connected layer
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_in_features * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_1, X_2):
        output_1 = self.resnet(X_1)
        output_2 = self.resnet(X_2)
        output = torch.cat((self.flatten(output_1), self.flatten(output_2)), 1)
        output = self.fc(output)
        return self.sigmoid(output)
