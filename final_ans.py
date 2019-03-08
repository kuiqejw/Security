import torch
import os, io
from torch import nn
from PIL import Image
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim
from scipy.misc import imread
from collections import OrderedDict
import torchvision
import skimage
from torchvision import transforms, utils
from torch.utils.data import random_split

class FlowerData(Dataset):
    def __init__(self, img_directory, label_file):
        self.data_path = []
        self.img_path = []
        self.targets = np.load(label_file)
        self.img_path = os.listdir(img_directory)
        self.transforms =  transforms.Compose([transforms.ToPILImage(),transforms.Resize(224),
            transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __getitem__(self, index):
        img_path= self.img_path[index]
        target = self.targets[index]
        img = skimage.io.imread("jpg/"+img_path)
        if(len(img.shape) == 2):
            img = np.stack((img,)*3, axis=-1)
        dic = {}
        dic['image'] = self.transforms(img)
        dic['class'] = target
        return dic

    def __len__(self):
        return len(self.img_path)


    def train_val_test_split(self, train_ratio, val_ratio):
        dataset_length = len(self.img_path)
        train_length = int(train_ratio * dataset_length)
        val_length = int(val_ratio * dataset_length)
        test_length = len(self) - train_length - val_length
        splits = [train_length, val_length, test_length]
        return random_split(self, splits)

dataset = FlowerData("C:\\Users\\ongajong\\Documents\\DeepLearning\\Week5\\jpg", 'labels.npy')
train_set, val_set, test_set = dataset.train_val_test_split(0.7, 0.1)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=0)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_losses = []
    for i, dict_of_values in enumerate(train_loader):
        data = torch.FloatTensor(dict_of_values["image"])
        data = data.to(device)
        target =  dict_of_values['class'].long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(epoch, "Epoch, ", index, "Index ", loss, "Loss: ")
    train_loss = torch.mean(torch.tensor(train_losses))
    print('\nEpoch: {}'.format(epoch))    
    return train_loss


def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for i, dict_of_values in enumerate(train_loader):
            data = torch.FloatTensor(dict_of_values["image"])
            data = data.to(device)
            target =  dict_of_values['class'].long().to(device)
            optimizer.zero_grad()
            output = model(data)
            # compute the batch loss
            batch_loss = nn.CrossEntropyLoss(output, target).item()
            val_loss += batch_loss
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(pred)):
                if pred[i] == target[i]:
                    correct+=1
                total+=1
            setcount+=1
            if setcount%10==0:
                print("Current Set: " + str(setcount))
        val_loss /= len(val_loader)
    
        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return val_loss

DATA_DIRECTORY = 'jpg/'
use_cuda = 0
batch_size = 32
num_epochs = 30
learning_rate = 1e-3

device = torch.device("cuda" if use_cuda else "cpu")
model = models.squeezenet1_1(pretrained=True)
# model.classifier[1] = nn.Conv2d(512, 102, kernel_size=(1,1), stride=(1,1))
# model_ft.num_classes = 102
model = models.squeezenet1_1(num_classes=102).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    val_loss = validate(model, device, val_loader)

    if (len(val_losses) > 0) and (val_loss < min(val_losses)):
        torch.save(model.state_dict(), "best_model.pt")
        print("Saving model (epoch {}) with lowest validation loss: {}"
              .format(epoch, val_loss))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

print("Training and validation complete.")

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(12,5))
epoch_list = np.arange(1, num_epochs+1)
plt.xticks(epoch_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epoch_list, train_losses, label="Training loss")
plt.plot(epoch_list, val_losses, label="Validation loss")
plt.legend(loc='upper right')
plt.show()

model.eval()

correct = 0
with torch.no_grad():
    for i, dic in enumerate(test_loader):
        data = dic['image'].to(device)
        labels = dic['class'].long().to(device)
        result = model(data)
        pred = result.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), 
                                                       100. * correct / len(test_loader.dataset)))