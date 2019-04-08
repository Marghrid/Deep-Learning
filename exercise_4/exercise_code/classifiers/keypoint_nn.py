import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4),    # Conv2d1
            nn.ELU(),                                                     # Activation1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),             # Maxpool2d1
            nn.Dropout(p=0.1),                                            # Dropout1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),    # Conv2d2
            nn.ELU(),                                                     # Activation2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),             # Maxpool2d2
            nn.Dropout(p=0.2),                                            # Dropout2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),   # Conv2d3
            nn.ELU(),                                                     # Activation3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),             # Maxpool2d3
            nn.Dropout(p=0.3),                                            # Dropout3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),  # Conv2d4
            nn.ELU(),                                                     # Activation4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),             # Maxpool2d4
            nn.Dropout(p=0.4)                                             # Dropout4
        )

# Dense1 Activation5 Dropout5 Dense2 Activation6 Dropout6 Dense3
        self.seq2 = nn.Sequential(
            nn.Linear(in_features=6400, out_features=1000),               # Dense1
            nn.ELU(),                                                     # Activation5
            nn.Dropout(p=0.5),                                            # Dropout5
            nn.Linear(in_features=1000, out_features=1000),               # Dense2
            nn.Linear(in_features=1000, out_features=1000),               # Activation6
            nn.Dropout(p=0.6),                                            # Dropout6
            nn.Linear(in_features=1000, out_features=30)                  # Dense3
        )
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.seq1(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.seq2(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
