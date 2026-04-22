import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)
        mask = torch.from_numpy(mask).long()

        return img, mask


def get_dataloaders(data_dir="data", batch_size=16):
    rgb_dir  = os.path.join(data_dir, "CameraRGB")
    mask_dir = os.path.join(data_dir, "CameraMask")

    image_paths = sorted([os.path.join(rgb_dir,  f) for f in os.listdir(rgb_dir)  if f.endswith('.png')])
    mask_paths  = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    print(f"Total samples: {len(image_paths)}")

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_dataset = CityscapesDataset(train_imgs, train_masks)
    test_dataset  = CityscapesDataset(test_imgs,  test_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, test_dataset, test_imgs, test_masks