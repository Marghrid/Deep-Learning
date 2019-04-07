"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        
        #VGG13
        # https://pytorch.org/docs/stable/torchvision/models
        print("vgg13")
        vgg_model = models.vgg13(pretrained = True)
        self.vgg13 = vgg_model.features

        # All pre-trained models expect input images normalized in the same way, 
        # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
        # where H and W are expected to be at least 224

        self.vgg13[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=100);
        
        print(self.vgg13)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=7)
        self.fc6.weight.data.mul_(0.001)
        self.fc6.bias.data.mul_(0.001)

        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout()

        self.fc7 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.fc7.weight.data.mul_(0.001)
        self.fc7.bias.data.mul_(0.001)

        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout()

        self.score_fr = nn.Conv2d(2048, num_classes, kernel_size=1, padding = 0)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
       
        x_size = x.size()

        x = self.vgg13(x)
        x = self.dropout6(self.relu6(self.fc6(x)))
        x = self.dropout7(self.relu7(self.fc7(x)))
        x = self.score_fr(x)

        # upsample layer. Must be defined here because needs to know size of input
        self.upsample = nn.Upsample(size = x_size[2:], mode = 'bilinear', align_corners=False)

        x = self.upsample(x)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
