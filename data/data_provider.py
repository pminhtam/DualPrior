import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import glob
import torchvision.transforms as transforms
import numpy as np
from .data_tools import sigma_estimate, random_augmentation, gaussian_kernel
from skimage import img_as_float32 as img_as_float
import random
##

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class SingleLoader(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir, gt_dir, image_size=512, radius=5, eps2=1e-6, noise_estimate=True):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.noise_path = []
        for files_ext in IMG_EXTENSIONS:
            self.noise_path.extend(glob.glob(self.noise_dir + "/**/*" + files_ext, recursive=True))
        self.gt_path = []
        for files_ext in IMG_EXTENSIONS:
            self.gt_path.extend(glob.glob(self.gt_dir + "/**/*" + files_ext, recursive=True))

        if len(self.noise_path) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.noise_dir + "\n"
                                                                                       "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.win = 2*radius + 1
        self.sigma_spatial = radius
        self.noise_estimate = noise_estimate
        self.eps2 = eps2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue
        """
        # rand_hflip = torch.rand(1)[0]
        # rand_vflip = torch.rand(1)[0]
        image_noise = np.array(Image.open(self.noise_path[index]).convert('RGB'))
        # name_image_gt = self.noise_path[index].split("/")[-1].replace("NOISY_", "GT_")
        # image_folder_name_gt = self.noise_path[index].split("/")[-2].replace("NOISY_", "GT_")
        # image_gt = Image.open(os.path.join(self.gt_dir, image_folder_name_gt, name_image_gt)).convert('RGB')
        image_gt = np.array(Image.open(self.noise_path[index].replace("Noisy",'Clean')).convert('RGB'))
        # image_gt = np.array(Image.open(self.noise_path[index].replace("noise",'gt').replace("NOISY_", "GT_")).convert('RGB'))
        # print(image_gt)
        H, W, C2 = image_gt.shape
        ind_H = random.randint(0, H-self.image_size)
        ind_W = random.randint(0, W-self.image_size)
        image_gt = self.crop_patch(image_gt,ind_H,ind_W)
        image_noise = self.crop_patch(image_noise,ind_H,ind_W)
        image_gt = img_as_float(image_gt)
        # print(image_gt)
        image_noise = img_as_float(image_noise)
        # data augmentation
        im_gt, im_noisy = random_augmentation(image_gt, image_noise)

        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))
        eps2 = torch.tensor([self.eps2], dtype=torch.float32).reshape((1,1,1))

        if self.noise_estimate:
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2, 0, 1)))
            return im_noisy, im_gt, sigma2_map_est, eps2
        else:
            return im_noisy, im_gt


    def __len__(self):
        return len(self.noise_path)
    def crop_patch(self, imgs_sets,ind_H,ind_W):
        im_gt = np.array(imgs_sets[ind_H:ind_H+self.image_size, ind_W:ind_W+self.image_size, :])
        return im_gt
IMG_EXTENSIONS_RAW = [
    '.RAW', '.raw','.MAT','.mat'
]
from utils.raw_util import read_raw
class SingleLoader_raw(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir, gt_dir, image_size=256):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.noise_path = []
        for files_ext in IMG_EXTENSIONS_RAW:
            self.noise_path.extend(glob.glob(self.noise_dir + "/**/*" + files_ext, recursive=True))
        self.gt_path = []
        for files_ext in IMG_EXTENSIONS_RAW:
            self.gt_path.extend(glob.glob(self.gt_dir + "/**/*" + files_ext, recursive=True))
        print(self.noise_path)
        if len(self.noise_path) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.noise_dir + "\n"
                                                                                       "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS_RAW)))

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue
        """
        image_noise = read_raw(self.noise_path[index])
        image_gt = read_raw(self.noise_path[index].replace("noise",'gt').replace("NOISY",'GT'))
        # image_gt = read_raw(self.noise_path[index])


        H, W, C2 = image_gt.shape
        ind_H = random.randint(0, H-self.image_size)
        ind_W = random.randint(0, W-self.image_size)
        image_gt = self.crop_patch(image_gt,ind_H,ind_W)
        image_noise = self.crop_patch(image_noise,ind_H,ind_W)
        image_gt = img_as_float(image_gt)
        image_noise = img_as_float(image_noise)

        im_gt, im_noisy = random_augmentation(image_gt, image_noise)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

    def __len__(self):
        return len(self.noise_path)
    def crop_patch(self, imgs_sets,ind_H,ind_W):
        im_gt = np.array(imgs_sets[ind_H:ind_H+self.image_size, ind_W:ind_W+self.image_size, :])
        return im_gt