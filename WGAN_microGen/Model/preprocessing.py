# import library
import imageio
import pandas as pd
import numpy as np
import os

import torch
from skimage import io, transform
from torchvision import transforms



class PET_MRI_Clinical_Dataset:

    def __init__(self, csv_file, root_dir_image, transform=None):
        """
        :param csv_file:  Hurst values
        :param root_dir_image: dir of images
        """
        self.H = pd.read_csv(csv_file)
        self.root_dir_img = root_dir_image
        self.transform = transform

    def __len__(self):
        return len(self.H)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.root_dir_img, str('image-')+self.H.iloc[idx, 3] + str('.png'))  # the first
        # col of the csv file should be the name
        h = self.H.iloc[idx, 1]  # consider the second column as the hurst value
        Image = imageio.imread(image_name)
        h = np.array([h])
        sample = {'image': Image, 'H': h}  # without Clinical data

        # return the values of the transformation
        if self.transform:
            sample = self.transform(sample)
        return sample


# resize and normalize the data

class Resize:
    """
    :param : sample of the data loader
    resize the images to 49 x 49
    """

    def __call__(self, sample):
        image,h = sample['image'],sample['H']
        img_resized = transform.resize(image, (49, 49))

        img_resized = np.expand_dims(img_resized, axis=0)

        img_normalized = (img_resized - np.min(img_resized)) / np.ptp(img_resized)

        return {'image': img_normalized, 'H': h}


class ToTensor:
    def __call__(self, sample):
        ###################################################################
        img, h = sample['image'], sample['H']
        ###################################################################



        return {'image': img, 'H': h}


def data_loader(batch_size):
    transformed_dataset = PET_MRI_Clinical_Dataset(csv_file='/home/bazanganif/Desktop/PhD/GAN_VAE/New folder/file.csv',
                                                   root_dir_image='/home/bazanganif/Desktop/PhD/GAN_VAE/New folder/data',
                                                   transform=transforms.Compose([Resize(),ToTensor(),]))
    data_loader = torch.utils.data.DataLoader(transformed_dataset, pin_memory=True, num_workers=4,
                                              batch_size=batch_size,
                                              shuffle=True)
    return data_loader


transformed_dataset = PET_MRI_Clinical_Dataset(csv_file='/home/bazanganif/Desktop/PhD/GAN_VAE/New folder/file.csv',
                                                   root_dir_image='/home/bazanganif/Desktop/PhD/GAN_VAE/New folder/data',
                                                   transform=transforms.Compose([Resize(),ToTensor(),]))
if __name__ == '__main__':
     batch_size = 40
     data_loader = data_loader(20)
     sample = transformed_dataset[10]
     print(" Image:", sample['image'].shape)
     print("Label",sample['H'])
