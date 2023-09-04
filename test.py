import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
from torch import optim
from item_config import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from custom_data import ItemClassifi
import torch.nn as nn
import os


# Data preparation
transform = A.Compose([
    A.Resize(640, 640),
    # A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    # A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


test_dataset = ItemClassifi("./add_dataset", transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
os.makedirs("./test_result", exist_ok=True)

def test_model(pth_epochs):
    # Initialize model and optimizer here (make sure it matches the architecture used during training)
    model = models.inception_v3(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Load checkpoint
    checkpoint = torch.load(f'./pth\\Inception3\\epoch_{pth_epochs}_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # If you also want to restore training metadata, you can load that here as well
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # Set to evaluation mode
    model.eval()

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device is ', device)
    model.to(device) 
    # Testing loop
    correct = 0
    total = 0
    d = test_dataset.label_dict
    print(d)

    result_path =f"./test_result\\epoch{pth_epochs}\\test_result_added_{pth_epochs}.txt"
    pred_path = f"./test_result\\epoch{pth_epochs}\\predicted.txt"

    os.makedirs(f"./test_result\\epoch{pth_epochs}", exist_ok=True)
    with open(result_path, 'w') as f:
        f.write("Predict, answer \n")
    with open(pred_path, 'w') as f:
        f.write("Predicted Vector")

    with torch.no_grad():
        for batch_i, set in enumerate(test_loader):
            images, labels = set[0].to(device), set[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(labels)
            # print(predicted)
            # print(test_dataset.label_dict[labels])
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            with open(pred_path, 'a') as f:
                f.write(f"{outputs.data}, {predicted}")
            with open(result_path+"_correct", 'w') as f:
                f.write("Predict, answer \n")
            with open(result_path+"_wrong", 'w') as f:
                f.write("Predict, answer \n")
            for i in range(predicted.size(dim=0)):
                idx = batch_i * 16 + i

                print(f"predicted : {d[predicted[i].item()]}, answer: {d[labels[i].item()]}")
                with open(result_path, 'a') as f:
                    f.write(f"[{idx}] predicted : {d[predicted[i].item()]}, answer: {d[labels[i].item()]}\n")
                if d[predicted[i].item()] == d[labels[i].item()]:
                    with open(result_path+"_correct", 'a') as f:
                        f.write(f"[{idx}] predicted : {d[predicted[i].item()]}, answer: {d[labels[i].item()]}\n")
                else:
                    with open(result_path+"_wrong", 'a') as f:
                        f.write(f"[{idx}] predicted : {d[predicted[i].item()]}, answer: {d[labels[i].item()]}\n")


    with open(result_path, 'a') as f:
        f.write(f'Accuracy: {100 * correct / total}%')

    print(f'Accuracy: {100 * correct / total}%')
    print(correct, total)


for i in range(1, 41):
    test_model(i*5)