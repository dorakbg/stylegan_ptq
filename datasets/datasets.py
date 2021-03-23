import glob
import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch


class PairedImageDataset(Dataset):
    def __init__(self, dataset_dir):
        '''
        Construct a dataset with all images from a dir.

        dataset: str. dataset name
        style: str. 'A2B' or 'B2A'
        '''
        transforms_ = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # (0,1) -> (-1,1)
        self.transform = transforms.Compose(transforms_)

        path_A = os.path.join(dataset_dir, 'distil_noise')
        path_B = os.path.join(dataset_dir, 'distil_pics')
        self.files_A = sorted(glob.glob(path_A + '/*.pth'))
        self.files_B = sorted(glob.glob(path_B + '/*.png'))

        # assert len(self.files_A) == len(self.files_B)

    def __getitem__(self, index):
        img_A = torch.load(self.files_A[index])[0]
        img_B = Image.open(self.files_B[index])
        img_B = img_B.convert("RGB")
        img_B = np.asarray(img_B)  # PIL.Image to np.ndarray
        img_B = Image.fromarray(np.uint8(img_B))  # np.ndarray to PIL.Image
        img_B = self.transform(img_B)
        return {'input': img_A, 'image': img_B}

    def __len__(self):
        return len(self.files_A)
