import argparse
import glob
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, utils
from torch.nn import functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
from sklearn.utils import shuffle
import segmentation_models_pytorch as smp
from PIL import Image, ImageFile
from dataloader import SIIMDataset
from Trainer import Trainer
from utils import ToTensor

if __name__=="__main__":
    seed= 12321
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main_path = "/home/mmaddur1/Segmentation/"
    root_dir_train='~/dataset/Pneumothorax_segmentation/train_jpeg/'
    root_dir_mask= '~/dataset/Pneumothorax_segmentation/train_jpeg/'

    root_dir_test='~/dataset/Pneumothorax_segmentation/test_jpeg/'
    root_dir_test_mask= '~/dataset/Pneumothorax_segmentation/test_jpeg/'

    with open('/home/mmaddur1/Segmentation/Pneumothorax-Segmentation-in-chest-X-rays/train/jpegs.txt', 'r') as f:
        train_path_list = f.read().split('\n')

    with open('/home/mmaddur1/Segmentation/Pneumothorax-Segmentation-in-chest-X-rays/train/masks.txt', 'r') as f:
        mask_path_list = f.read().split('\n')

    with open('/home/mmaddur1/Segmentation/Pneumothorax-Segmentation-in-chest-X-rays/test/jpegs.txt', 'r') as f:
        test_path_list = f.read().split('\n')

    with open('/home/mmaddur1/Segmentation/Pneumothorax-Segmentation-in-chest-X-rays/test/masks.txt', 'r') as f:
        test_mask_path_list = f.read().split('\n')

    
    batch_size = 16
    img_size = 128
    transformed_dataset = SIIMDataset(train_path_list, mask_path_list, root_dir_mask, root_dir_train, img_size,validate=False,transform=transforms.Compose([ToTensor()]))
    validation_dataset = SIIMDataset(test_path_list, test_mask_path_list, root_dir_test_mask, root_dir_test, img_size,validate=True,transform=transforms.Compose([ToTensor()]))

    train_dataloader = DataLoader(transformed_dataset, batch_size=batch_size,num_workers=4,shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size,num_workers=4,shuffle=True)
    model = smp.UnetPlusPlus("efficientnet-b5", encoder_weights=None, activation=None)
    log_directory = '/home/mmaddur1/Segmentation/Pneumothorax-Segmentation-in-chest-X-rays/' + "runs/prediction"
    
    sol = Trainer(model, train_dataloader , val_dataloader,  use_cuda = True , logdir = log_directory , lr=5e-4)
    sol.train(25,main_path)






