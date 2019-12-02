import pandas as pd
import numpy as np
import os, torch, random
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from skimage import io
from math import floor, ceil
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 3)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

#Pobieranie zbiorów 

trainDir = "data/TRAIN"
testDir = "data/TEST"
trainDirs = ["data/apples/","data/bananas/","data/pears/"]
labels = []
filePaths = []

i=0
x=0

for dir in trainDirs:
    l = os.listdir(dir)
    n = len(l)
    filePaths =  filePaths + [dir + id for id in l]
    labels = labels + [i for j in range(x,n+x)]
    x = x + n 
    i = i + 1
    print(n)

nl = os.listdir(testDir)
numberOfTestImgs = len(nl)
#print(numberOfTestImgs)
#print(filePaths)
#print(labels)

#Mieszanie zbiorów aby nie były po kolei klasami
combined = list(zip(filePaths, labels))
random.shuffle(combined)
random.shuffle(combined)
filePaths, labels = zip(*combined)
#print(filePaths)
#print(labels)

#Trenowanie
train_img = []
for path in filePaths:
    img = io.imread(path, as_gray=True)
    img /= 255.0
    img.astype('float32')
    train_img.append(img)

train_x = np.array(train_img)
train_y = labels

print(train_x.shape)

if x%10 != 0:
    x = int(x - x%10)

train_x = train_x[:x]
train_y = train_y[:x]

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1)

train_x = train_x.reshape(x - int(0.1*x), 1, 30, 30)
train_x = torch.from_numpy(train_x)

train_y = np.asarray(train_y).astype(int)
train_y = torch.from_numpy(train_y)

val_x = val_x.reshape(int(0.1*x),1,30,30)
val_x = torch.from_numpy(val_x)

val_y = np.asarray(val_y).astype(int)
val_y = torch.from_numpy(val_y)

print(val_x.shape)
print(val_y.shape)

# defining model

model = Net()

optimizer = Adam(model.parameters(), lr=0.07)

criterion = CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

n_epochs = 25
train_losses = []
val_losses = []

def train(epoch):
    global model
    model.train()
    tr_loss = 0
    x_train, y_train = Variable(train_x), Variable(train_y)
    x_val, y_val = Variable(val_x), Variable(val_y)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    
    optimizer.zero_grad()

    model=model.double()
    output_train = model(x_train)
    output_val = model(x_val)

    loss_train = criterion(output_train, y_train.long())
    loss_val = criterion(output_val, y_val.long())
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

for epoch in range(n_epochs):
    train(epoch)

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

#Walidacja

with torch.no_grad():
    output = model(train_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

print(accuracy_score(train_y, predictions))

#Testowanie


test_img = []
for filePath in os.listdir("data/TEST/"):
    img = io.imread("data/TEST/"+filePath, as_gray=True)
    img /= 255.0
    img.astype('float32')
    test_img.append(img)

test_x = np.array(test_img)
test_x = test_x.reshape(numberOfTestImgs, 1, 30, 30)
test_x = torch.from_numpy(test_x)

with torch.no_grad():
    output = model(test_x.cuda())

testLabels = []
i = 0
for fruit in trainDirs:
    testLabels = testLabels + [i for j in range(0,15)]
    i += 1


softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

print(predictions)
print(testLabels)
goodPredictions = 0

for i in range(0, len(predictions)):
    if(predictions[i]==testLabels[i]):
        goodPredictions += 1

print("Correct predictions: " + str(goodPredictions))
print("Percentage of successful predictions: " + str(100*goodPredictions/len(predictions))+"%")
