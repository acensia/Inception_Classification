import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.models import inception_v3
from resnest.torch import resnest50
from custom_data import ItemClassifi
# from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import json
import os
from item_config import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device is ', device)
os.makedirs("./pth_added", exist_ok=True)
 

# Data preparation
train_transform = A.Compose([
    A.Resize(640, 640),
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


val_transform = A.Compose([
    A.Resize(height=640, width=640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

 
dataset = ItemClassifi("./add_dataset")
data_len = len(dataset)
print(data_len)
train_len = int(data_len * 0.9)
trainset, valset = random_split(dataset, [train_len, data_len - train_len])
trainset.dataset.transform = train_transform
valset.dataset.transform = val_transform

# valset, testset = random_split(dataset, [173, 173])


trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


# Model preparation
model = inception_v3(pretrained=True)
# model = torchvision.models.mobilenet_v2(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  
model = model.to(device)
 

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# check point
checkpoint = torch.load('./pth/Inception3/epoch_120_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("Weight loaded from 120 epoch")

best_val_acc = 0.0
train_losses = []
val_losses = []
train_acc = []
val_acc = [] 
best_acc_ep = 0

print("Start Learning....")
print(f"Model Name : {model.__class__.__name__}")
os.makedirs(f"./pth_added/{model.__class__.__name__}", exist_ok=True)
with open(f"./pth_added/{model.__class__.__name__}/{model.__class__.__name__}_{epoch_start}_{epoch_end}.txt", 'w') as f:
   f.write("Epoch, Train_Loss, Val_Loss, Val_Acc")
for epoch in range(epoch_start, epoch_end):  # loop over the dataset multiple times
    # Training
    model.train()
    start_time = time.time()
    for i, data_box in enumerate(trainloader):
        data, target = data_box[0].to(device), data_box[1].to(device)

        optimizer.zero_grad()
        outputs, aux_outputs = model(data)
        loss1 = criterion(outputs, target)
        loss2 = criterion(aux_outputs, target)
        loss = loss1 + 0.4 * loss2  # As per the paper regarding Inception architecture
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data_box in valloader:
            data, target = data_box[0].to(device), data_box[1].to(device)
            outputs = model(data)
            val_loss += criterion(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(valset)
    val_losses.append(val_loss)
    curr_val_acc= 100*correct/len(valset)
    val_acc.append(curr_val_acc)
    if best_val_acc < curr_val_acc:
        best_val_acc = curr_val_acc
        best_acc_ep = epoch+1
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}, Validation Accuracy: {curr_val_acc}%")
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds")

    # Save the model's state dictionary after a specific epoch
    if (epoch + 1) % 5 == 0:  # Define save_interval as desired
        checkpoint_path = f'./pth_added/{model.__class__.__name__}/epoch_{epoch+1}_checkpoint.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_acc
        }, checkpoint_path)
        with open(f"./pth_added/{model.__class__.__name__}/{model.__class__.__name__}_{epoch_start}_{epoch_end}.txt", 'a') as f:
            f.write(f"{epoch+1}, {train_losses}, {val_losses}, {val_acc}")
            train_losses = []
            val_losses = []
            val_acc = []

 

print(f'Accuracy on test images: {best_val_acc}%, in epoch : {best_acc_ep}')
with open(f"./pth_added/{model.__class__.__name__}/{model.__class__.__name__}_{epoch_start}_{epoch_end}.txt", 'a') as f:
    f.write(f'Accuracy on test images: {best_val_acc}%, in epoch : {best_acc_ep}')

