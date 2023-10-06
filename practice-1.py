# import the package which we will use to programing
import os
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import math
import multiprocessing
# --------------------------------------------------------------------------------------------------------------------------------------------------
from PIL import Image
# from google.colab import drive
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets
# --------------------------------------------------------------------------------------------------------------------------------------------------
# use gpu if you have
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Use device:",device)

# set random seed
SEED = 6220
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
multiprocessing.set_start_method("fork")
# --------------------------------------------------------------------------------------------------------------------------------------------------
train_transform = transforms.Compose([

    ##############################################################################
    #                    TODO: Write the transform functions                     #
    ##############################################################################

    ##### Try to apply the augmentation functions
    transforms.RandomHorizontalFlip(),
    ## transforms.RandomRotation(15),

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    transforms.ToTensor(),

    ##############################################################################
    #                    TODO: Write the normalized functions                    #
    ##############################################################################

    ##### Try to apply the normalized functions
    transforms.Normalize(0.485, 0.229),

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    ])

valid_transform = transforms.Compose([

    transforms.ToTensor(),

    ##############################################################################
    #                    TODO: Write the normalized functions                    #
    ##############################################################################

    ##### Try to apply the normalized functions
    transforms.Normalize(0.485, 0.229),

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    ])
# --------------------------------------------------------------------------------------------------------------------------------------------------
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)

length = len(trainset)
print('Number of total train dataset:', length)

# (Bonus Part)
##############################################################################
#                    TODO: Split the training and validation sets            #
##############################################################################


##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

testset = datasets.MNIST(root='./data', train=False, download=True, transform=valid_transform)

print('Number of testset:', len(testset))
class_num = 10
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# --------------------------------------------------------------------------------------------------------------------------------------------------
# Loaded Datasets to DataLoaders

##############################################################################
#                    TODO: Validation Dataloader                             #
##############################################################################

# please change the batch_size
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers = 2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers = 2)

# (Bonus Part) Add a validation dataloader here.

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################
# --------------------------------------------------------------------------------------------------------------------------------------------------
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(32)))
# --------------------------------------------------------------------------------------------------------------------------------------------------
##############################################
# Build your model here!
#
# Practice:
#   Try to implement MLP with pytorch !
##############################################

class trainmodel(nn.Module):
    def __init__(self):
        super(trainmodel, self).__init__()

        self.feat_classifier = nn.Sequential(

            ##############################################################################
            #                    TODO: Complete the code                                 #
            ##############################################################################

            # The Input should be height * width * channel
            # nn.Linear(height * width * channel, # of hidden layer neurons),
            nn.Linear(28*28*1,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.Linear(256,64),
            nn.Linear(64,class_num),
            # (Opt) Try to make more different here

            # The output should be as class_num
            # nn.Linear(# of hidden layer neurons, class_num),

            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################

            nn.Softmax(dim=1),
            )

    def forward(self, x):

        ##############################################################################
        #                    TODO: Complete the code.                                #
        ##############################################################################

        # The shape of flatten output should be height * width * channel
        # x = x.view(-1, height * width * channel)
        x = x.view(-1, 28*28*1)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

        out = self.feat_classifier(x)
        return out

model = trainmodel()
model.to(device)
# --------------------------------------------------------------------------------------------------------------------------------------------------
##############################################################################
#                          TODO: Fill the parameters                         #
##############################################################################

batch_size = 128
channel = 1
height = 28
width = 28

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

result = model(torch.rand((batch_size, channel, height, width)).to(device))
print(result.shape)
# --------------------------------------------------------------------------------------------------------------------------------------------------
## Use GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
net = model.to(device)
criterion = nn.CrossEntropyLoss()

##############################################################################
#                         TODO: Design the Parameters                        #
##############################################################################

learning_rate = 0.008
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
epochs = 10

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

net.train()
# --------------------------------------------------------------------------------------------------------------------------------------------------
# Training model
train_loss = []
train_acc = []

for epoch in tqdm(range(epochs)):# loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
##        if(i==5):
 ##         break
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        train_loss.append(loss.cpu().item())
        _, prex= torch.max(outputs, 1)
        metric = MulticlassAccuracy(num_classes = class_num)
        train_acc.append(metric(prex.cpu(), labels.cpu()))

        # backward (compute gradient)
        loss.backward()

        # optimize (update weights)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i + 1) % 100 == 0:    # print every 100 iterations
            print('[Epoch: %d, Iteration: %5d] Loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # (Bonus part)
    ##############################################################################
    #                   TODO: Compute Validation Loss and Accuracy               #
    ##############################################################################

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

print('Finished Training')
# --------------------------------------------------------------------------------------------------------------------------------------------------
def plt_loss_acc(list_to_draw,name):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    if name=="loss":
      ax1.set_title('Loss')
      ax1.plot(list_to_draw)
    elif name=="acc":
      ax1.set_title('Accuracy')
      ax1.plot(list_to_draw)
    ax1.set_xlabel('epoch')
    ax1.legend(['train_loss', 'train_acc'], loc='upper left')
    plt.show()

plt_loss_acc(train_loss, "loss")
plt_loss_acc(train_acc, "acc")
# --------------------------------------------------------------------------------------------------------------------------------------------------
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))