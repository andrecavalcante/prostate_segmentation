from __future__ import print_function
import torch 
import sys
import time
import torch.nn as nn
import torchvision
from torchvision import utils, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
from metrics import pixel_acc, iou2d, iou3d
from unet import UNet
from Prostate_loader import ProstateDataset, Resize, ToTensor, Normalize, Rotate
import os
import numpy as np

# Root directory for the project
root_dir = ""

# .csv files for dataset
train_file = os.path.join(root_dir, 'train.csv')
val_file = os.path.join(root_dir, 'val.csv')
test_file = os.path.join(root_dir, 'test.csv')

# Model evaluation function
def eval(model, data_loader, device):
    model.eval() # Toggle model into eval mode because of batchnorm module
    with torch.no_grad():
        pred_volume = torch.tensor([]).long().to(device)
        target_volume = torch.tensor([]).long().to(device)
        iou_slice = [] 
        for batch_test in data_loader:
            image = Variable(batch_test['image'])
            label = Variable(batch_test['label'])
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
        # iou on MR slice
            iou_slice.append(iou2d(predicted, label, cls=1)) # calculates iou for each slice
        val_iou = np.nanmean(np.array(iou_slice)) # calculates mean slicewise iou
        # iou on MR volume   
           #pred_volume = torch.cat((pred_volume, predicted.unsqueeze(0)),0) # form predicted volume by concat slices
           #target_volume = torch.cat((target_volume, label.unsqueeze(0)),0) # form target volume   
        #val_iou = iou3d(pred_volume, target_volume, cls=1) # calculates iou considering the entire 3d volume
        
        #print iou
        print("Validation IoU: {:.2f}".format(val_iou))

# Model training function        
def train(model, optimizer, criterion,
          train_loader, val_loader, device, num_epochs):
   
    model.train() # Toggle model into train mode because of batchnorm module
    step_size = 100
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):

            optimizer.zero_grad()

            images = Variable(batch['image'])
            labels = Variable(batch['label'])
            images = images.to(device)
            labels = labels.to(device)
        
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
        
            # Backward and optimize
            loss.backward()
            optimizer.step()
        
            if (i+1) % step_size == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.16f}' 
                   .format(epoch+1, num_epochs, 
                   i+1, total_step, 100* loss.item()))
        # Validation        
        eval(model, val_loader, device)

# main function        
def main():

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Hyper parameters
    num_epochs = 4000
    num_classes = 2
    step_size = 100
    batch_size = 1
    learning_rate = 3e-5

    transform = transforms.Compose([Normalize(), ToTensor()])

    train_dataset = ProstateDataset(csv_file=train_file,
                                    root_dir=root_dir,
                                    phase='train',
                                    transform=transform)

    val_dataset = ProstateDataset(csv_file=val_file,
                                  root_dir=root_dir, 
                                  phase='val', 
                                  transform=transform)

    test_dataset = ProstateDataset(csv_file=test_file, 
                                   root_dir=root_dir,
                                   phase='test', 
                                   transform=transform)

    train_loader = torch.utils.data.DataLoader(
                                   dataset=train_dataset,
                                   batch_size=batch_size, 
                                   shuffle=True)

    val_loader = torch.utils.data.DataLoader(
                                   dataset=val_dataset,
                                   batch_size=1,  
                                   shuffle=False)

    test_loader = torch.utils.data.DataLoader(
                                   dataset=test_dataset,
                                   batch_size=1,  
                                   shuffle=False)

    model = UNet(n_channels=1, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                              lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train(model, optimizer, criterion, train_loader, val_loader, device, num_epochs)
    

if __name__ == "__main__":
    main()
