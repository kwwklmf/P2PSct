import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# 禁用albumentations的版本检查
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "snap/snapd-desktop-integration/253/Documents/P2PSct/data_all/maps/train"
VAL_DIR = "snap/snapd-desktop-integration/253/Documents/P2PSct/data_all/maps/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 256
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)