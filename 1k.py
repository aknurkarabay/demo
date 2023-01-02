from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy
import pandas as pd
import pickle
import csv
from efficientnet_pytorch import EfficientNet

feature_extract = False
model_list = ['squeezenet','resnet', 'densenet', 'efficientnet']

data_dir = '../1k'

model_name = model_list[3]
learning_rate = 0.001
epoch_number = 40
batch = 68
learning_rate_scheduler = True
cudnn.benchmark = True
plt.ion()   # interactive mode

# LOAD THE DATA ##################
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(380),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(380),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch,
                                            shuffle=True, num_workers=4)
            for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:2") #if torch.cuda.is_available() else "cpu")
########################################

results = {'train_loss':[], 'val_loss':[], 'train_accuracy':[], 'val_accuracy':[]}
# TRAINING THE MODEL
train_loss = []
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):

    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase=='train':
                results['train_loss'].append(epoch_loss)
                results['train_accuracy'].append(epoch_acc)
            if phase=='val':
                results['val_loss'].append(epoch_loss)
                results['val_accuracy'].append(epoch_acc)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

########################################

# FINETUNE THE CONVNET

if model_name == "squeezenet":
    model_ft = models.squeezenet1_0(pretrained=True)
    model_ft.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = len(class_names)
elif model_name == "resnet":
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
elif model_name == "densenet":
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, len(class_names)) 
elif model_name == "efficientnet":
    model_ft = EfficientNet.from_pretrained('efficientnet-b4')
    
    # To use the custom weights of the previouls trained model
    # net_weight = 'pretrained_custom.pt'
    # state_dict = torch.load(net_weight)
    # model_ft.load_state_dict(state_dict)

    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, len(class_names))
else:
    print('invalid model, exiting...')
    exit()

if feature_extract == True:
    for param in model_ft.parameters():
        param.requires_grad = False

else:
    layer = 0
    for child in model_ft.children():
        layer += 1
        if layer < 7:
            for param in child.parameters():
                param.requires_grad = False

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
if learning_rate_scheduler==True:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=epoch_number)
else:
    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=epoch_number)
PATH = "results/1k_{}_b4.pt".format(model_name)

# Save
torch.save(model_ft, PATH)

# Plotting the train and validation loss curves
plt.style.use("ggplot")
plt.figure()
plt.plot(results['train_loss'], label="train_loss")
plt.plot(results['val_loss'], label="val_loss")
plt.title("Training and validation lossess")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig('results/loss-curve_food1k_{}.png'.format(model_name)) 



