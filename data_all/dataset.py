from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import config

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Filter out hidden files and ensure we only have image files
        self.list_files = [f for f in os.listdir(self.root_dir) 
                              if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]#分两半

        augmentations = config.both_transform(image = input_image, image0 = target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]
        input_image = config.transform_only_input(image = input_image)["image"]
        target_image = config.transform_only_mask(image = target_image)["image"]

        return input_image, target_image
    

    
    