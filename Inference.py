from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar

import pandas as pd

import medicalDataLoader
from UNet import *
from utils import *
import sys

import time

# CUDA_VISIBLE_DEVICES=6 python Inference.py ./model/Best_UNetG_Dilated_Progressive.pkl Best_UNetG_Dilated_Progressive_Inference
def runInference(argv):
    print("-" * 40)
    print("~~~~~~~~  Starting the inference... ~~~~~~")
    print("-" * 40)

    batch_size_val = 1
    batch_size_val_save = 1
    batch_size_val_savePng = 1

    transform = transforms.Compose([transforms.ToTensor()])

    mask_transform = transforms.Compose([transforms.ToTensor()])

    root_dir = "../DHCP-R2/pruned DHCP - release 2"
    modelName = "UNet3D"
    modality = "T2"
    num_classes = 10

    df = pd.read_csv("DHCP-test-dataframe-with-patches.csv")

    print("...Loading model...")
    try:
        netG = torch.load(modelName)
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass

    netG.cuda()


if __name__ == "__main__":
    runInference(sys.argv[1:])
