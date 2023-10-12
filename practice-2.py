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

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# use gpu if you have
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print ("MPS device is available. Successfully initiated:")
    print (x)
    device = mps_device
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU is available.")
    print("GPU device count:", torch.cuda.device_count())
    print("Current GPU device:", torch.cuda.current_device())
    print("GPU device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = torch.device("cpu")

print("Use device:",device)

# set random seed
SEED = 6220
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
multiprocessing.set_start_method("fork")
# -------------------------------------------------------------------------------------------------------------------------------------------------------
train_transform = transforms.Compose([

    ##############################################################################
    #                    TODO: Write the transform functions                     #
    ##############################################################################

    ##### Try to apply the augmentation functions

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    transforms.ToTensor(),

    ##############################################################################
    #                    TODO: Write the normalized functions                    #
    ##############################################################################

    ##### Try to apply the normalized functions
    transforms.Normalize(0.485, 0.229)

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
    transforms.Normalize(0.485, 0.229)

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    ])
# -------------------------------------------------------------------------------------------------------------------------------------------------------
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=valid_transform)

print('Number of total training dataset:', len(trainset))
print('Number of testing dataset:', len(test_set))

length = len(trainset)
n_TrainData = math.floor(length * 0.8)
n_ValidData = length - n_TrainData
print('Number of training data : ',n_TrainData)
print('Number of validation data : ', n_ValidData)

train_set, valid_set = torch.utils.data.random_split(
    trainset,
    [n_TrainData, n_ValidData],
    generator=torch.Generator().manual_seed(0)
)


class_num = 10
classes = ('T-shirt/top', ' Trouser', ' Pullover', ' Dress',
           ' Coat', ' Sandal', ' Shirt', ' Sneaker', ' Bag', 'Ankle boot')
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# Loaded Datasets to DataLoaders

##############################################################################
#                    TODO: Validation Dataloader                             #
##############################################################################

# please change the batch_size
trainloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers = 0)
validloader = DataLoader(valid_set, batch_size=128, shuffle=False, num_workers = 0)
testloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers = 0)

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################
# -------------------------------------------------------------------------------------------------------------------------------------------------------
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
print(' '.join('%5s' % classes[labels[j]] for j in range(2)))
# -------------------------------------------------------------------------------------------------------------------------------------------------------
##############################################
# Build your model here!
#
# Practice:
#   Try to implement MLP with pytorch !
##############################################

class trainmodel(nn.Module):
    def __init__(self):
        super(trainmodel, self).__init__()


        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

        )

        self.feat_classifier = nn.Sequential(

            ##############################################################################
            #                    TODO: Complete the code                                 #
            ##############################################################################

            nn.Linear(in_features=16*5*5, out_features=224),
            nn.ReLU(),
            nn.Linear(in_features=224, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################

            nn.Softmax(dim=1)
        )

    def forward(self, x):

        ##############################################################################
        #                    TODO: Complete the code.                                #
        ##############################################################################

        # The shape of flatten output should be height * width * channel
        # x = x.view(-1, height * width * channel)

        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

        out = self.feat_classifier(x)
        return out

model = trainmodel()
model.to(device)
# -------------------------------------------------------------------------------------------------------------------------------------------------------
x = torch.rand(1,28,28).to(device)
out = model.conv_block(x)
# flat = model.forward(x)
out.shape
# -------------------------------------------------------------------------------------------------------------------------------------------------------
##############################################################################
#                          TODO: Fill the parameters                         #
##############################################################################

batch_size = 64
channel = 1
height = 28
width = 28

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

result = model(torch.rand((batch_size, channel, height, width)).to(device))
print(result.shape)
# -------------------------------------------------------------------------------------------------------------------------------------------------------
## Use GPU
print(device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

##############################################################################
#                         TODO: Design the Parameters                        #
##############################################################################

learning_rate = 0.0002
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 20

##############################################################################
#                              END OF YOUR CODE                              #
##############################################################################

model.train()
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# Training model
train_loss_list = []
train_acc_list = []
valid_loss_list = []
valid_acc_list = []

# Specify the saving path
SAVING_PATH = '/.'

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, epochs+1):# loop over the dataset multiple times

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print('running epoch: {}'.format(epoch))

    # train the model
    model.train()
    train_correct = 0
    train_total = 0
    for data, target in tqdm(trainloader):
      # move tensors to GPU if CUDA is available
      data, target = data.to(device), target.to(device)
      # clear the gradients of all optimized variables
      optimizer.zero_grad()
      # forward pass: compute predicted outputs by passing inputs to the model
      output = model(data)
      # calculate the batch loss
      loss = criterion(output, target)
      # backward pass: compute gradient of the loss with respect to model parameters
      loss.backward()
      # perform a single optimization step (parameter update)
      optimizer.step()
      # update training loss
      train_loss += loss.item()*data.size(0)
      # update training Accuracy
      train_total += target.size(0)
      _, predicted = torch.max(output.data, 1)
      train_correct += (predicted == target).sum().item()


    # validate the model
    model.eval()
    valid_correct = 0
    valid_total = 0
    for data, target in tqdm(validloader):
        # move tensors to GPU if CUDA is available
        target = target.long()
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item()*data.size(0)
        # update validation Accuracy
        valid_total += target.size(0)
        _, predicted = torch.max(output.data, 1)
        valid_correct += (predicted == target).sum().item()
    # calculate average losses
    train_loss = train_loss/len(trainloader.dataset)
    valid_loss = valid_loss/len(validloader.dataset)

    # print training/validation statistics
    print('Training Loss: {:.6f} \tTraining Accuracy: {:.6f}'.format(train_loss,(100 * train_correct / train_total)))
    print('Validation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(valid_loss,(100 * valid_correct / valid_total)))

    train_loss_list.append(train_loss)
    train_acc_list.append(100 * train_correct / train_total)
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(100 * valid_correct / valid_total)

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        # torch.save(model.state_dict(), SAVING_PATH+'/model_weight.pth')
        valid_loss_min = valid_loss

print('Finished Training')
# -------------------------------------------------------------------------------------------------------------------------------------------------------
def plt_loss_acc(list_to_draw,name):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    if name=="train_loss":
      ax1.set_title('Train Loss')
      ax1.plot(list_to_draw)
    elif name=="train_acc":
      ax1.set_title('Train Accuracy')
      ax1.plot(list_to_draw)
    elif name=="valid_loss":
      ax1.set_title('Valid Loss')
      ax1.plot(list_to_draw)
    elif name=="valid_acc":
      ax1.set_title('Valid Accuracy')
      ax1.plot(list_to_draw)

    ax1.set_xlabel('epoch')
    plt.show()
# -------------------------------------------------------------------------------------------------------------------------------------------------------
plt_loss_acc(train_loss_list, "train_loss")
plt_loss_acc(train_acc_list, "train_acc")
plt_loss_acc(valid_loss_list, "valid_loss")
plt_loss_acc(valid_acc_list, "valid_acc")
# -------------------------------------------------------------------------------------------------------------------------------------------------------
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))