from torch.utils.data import Dataset
import os
import glob
import cv2
# from PIL import Image
import numpy as np
import albumentations as A

class ItemClassifi(Dataset):
    def __init__(self, data_path, is_window=True, transforms=None):
        slash = "\\"
        if not is_window:
            slash = "/"
        label_path = glob.glob(os.path.join(data_path, "*"))

        self.data_list = []
        self.label_list = []
        self.transform = transforms
        label_dict = {}

        for i, label_folder in enumerate(label_path):
            label_name = os.path.basename(label_folder)
            label_dict[label_name] = i
            img_files = glob.glob(os.path.join(label_folder, "*.png"))
            img_files += glob.glob(os.path.join(label_folder, "*.[j][p]*[g]"))
            for file in img_files:
                self.data_list.append(file)
                self.label_list.append(i)
            
    def __getitem__(self, idx):
        with open(self.data_list[idx],'rb') as f:
            byte_array = bytearray(f.read())
        img = cv2.imdecode(np.asarray(byte_array, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (640, 640))
        # image = Image.fromarray(img)
        if self.transform:
            augmented = self.transform(image= img)
            img = augmented['image']
        return img, self.label_list[idx]

    def __len__(self) :
        return len(self.label_list)            