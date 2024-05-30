import torch.nn as nn
import torch.nn.functional as f

#50x50 pixels
img_size = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Convolutional Layers
        #last value has to be the first in the following layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        #Fully Connected Layer
        self.fc1 = nn.Linear(256 * 3 * 3, 512)

        #Final Fully Connected Layer has 2 in the final parameter
        #due to this neural network being a binary classifier (benign ou malignant)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    #Method has to be called "forward"
    #How the image "moves through" the network
    def forward(self, x):
        #RELU = Linear Regression: If num is positive, the number stays; if num is negative, makes it 0
        #Max Pooling 2d = transforms the output into a "smaller output". The (2,2) is the kernel size.
        x = f.max_pool2d(f.relu(self.conv1(x)), (2,2))
        x = f.max_pool2d(f.relu(self.conv2(x)), (2,2))
        x = f.max_pool2d(f.relu(self.conv3(x)), (2,2))
        x = f.max_pool2d(f.relu(self.conv4(x)), (2,2))

        #"Flatten" the values after conv4
        x = x.view(-1, 256 * 3 * 3)

        #Pass through fc1
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        #Pass through fc2
        x = self.fc2(x)

        x = f.log_softmax(x, dim=1)

        return (x)
