from __future__ import print_function, division
import os
from random import shuffle
from pandas.core.algorithms import isin
import torch
import pandas as pd
from skimage import io, transform
import numpy as np

# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import nibabel as nib
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
import pdb

warnings.filterwarnings("ignore")


def make_3Ddataset(
    root_dir,
    df_root_path,
    mode,
    modality,
    num_classes,
    labelled_if_train=False,
    seed=0,
    labelled_samples=0,
    val_samples=0,
    dataset="DHCP",
):
    """
    Args:
        root_dir : path to volumes
        df_root_path (string): dataframe directory containing csv files
        mode (string): 'train', 'val', 'test'
        num_classes : 10 or 88 for DHCP data
    """
    assert mode in ["train", "val", "test", "unl_train"]
    items = []

    # df = pd.read_csv(f'{df_root_path}/DHCP-{mode}-dataframe-with-patches.csv')
    #    df = pd.read_csv(f'{df_root_path}/DHCP-patches-{mode}-dataframe.csv')

    if dataset == "WHS":
        if mode == "train" and labelled_if_train == False:
            mode = "unl_train"
        df = pd.read_csv(f"{df_root_path}/WHS-patches-{mode}-dataframe.csv")
        df = df.query(
            f"{modality}_modality == {modality}_modality"
        )  # those who have selected modality data

        if mode == "train":
            subs = np.unique(df["sub"].astype("int").values)
            # subs = pd.from_numpy(subs)
            if labelled_samples == len(subs):
                subs_labelled = subs
            elif labelled_samples > len(subs):
                assert False, "There are not enough labelled samples"
            else:
                subs_labelled, _ = train_test_split(
                    subs, train_size=labelled_samples, random_state=seed, shuffle=True
                )
            if labelled_if_train:
                df = df.loc[df["sub"].astype("int").isin(subs_labelled)]

            print(f"# of data for {mode} is : {len(df)}")
        elif mode == "val":
            df = pd.read_csv(f"{df_root_path}/WHS-patches-{mode}-dataframe.csv")

    elif dataset == "DHCP":
        df = pd.read_csv(f"{df_root_path}/DHCP-patches-{mode}-dataframe.csv")
        df = df.query(
            f"{modality}_modality == {modality}_modality"
        )  # those who have selected modality data
        if mode == "train":
            # for n,j in enumerate(df['ses'].values):
            #     if type(j) != int:
            #         df['ses'].loc[n] = j.replace('d', '')
            sess = np.unique(df["ses"].astype("int").values)
            print("TOTAL SAMPLES :", len(sess))
            # sess = pd.from_numpy(sess)
            sess_labelled, sess_unlabelled = train_test_split(
                sess, train_size=labelled_samples, random_state=seed, shuffle=True
            )
            if labelled_if_train:
                df = df.loc[df["ses"].astype("int").isin(sess_labelled)]
            else:
                df = df.loc[df["ses"].astype("int").isin(sess_unlabelled)]
            add = "labelled" if labelled_if_train else "unlabelled"
            print(f"# of data for {add} {mode} is : {len(df)}")
        elif mode == "val" and val_samples > 0:
            subs = np.unique(df["sub"].values)
            val, _ = train_test_split(
                subs, train_size=val_samples, random_state=seed, shuffle=True
            )
            df = df.loc[df["sub"].isin(val)]
            print(f"# of data for {mode} is : {len(df)}")
        else:
            print(f"# of data for {mode} is : {len(df)}")
    data_paths = df[f"{modality}_modality"].values
    GT_paths = df[f"segmentation_{num_classes-1}classes"].values
    mask_paths = df["brainmask_bet"].values

    subs = df["sub"].values
    if dataset == "DHCP":
        sess = df["ses"].values
        names = [subs[i] + "-" + str(sess[i]) for i in range(len(df))]
    else:
        names = [subs[i] for i in range(len(df))]

    patches = df["patches"].values

    for it_im, it_mk, it_gt, it_patch, it_nm in zip(
        data_paths, mask_paths, GT_paths, patches, names
    ):
        item = (
            os.path.join(root_dir, it_im),
            os.path.join(root_dir, it_mk),
            os.path.join(root_dir, it_gt),
            eval(it_patch),
            it_nm,
        )
        items.append(item)

    return items


def normalize_intensity(
    img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)
):
    """
    Accept the image tensor and normalizes it (ref: MedicalZooPytorch)
    Args:
        img_tensor (tensor): image tensor
        normalization (string): choices = "max", "mean"
        norm_values (array): (MEAN, STD, MAX, MIN)

    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / (std_val + 1e-10)
    elif normalization == "max":
        # max_val, _ = torch.max(img_tensor)
        # img_tensor = img_tensor / max_val
        img_tensor = img_tensor / img_tensor()
    elif normalization == "brats":
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0.0, img_tensor, normalized_tensor)
        final_tensor = (
            100.0
            * (
                (final_tensor.clone() - norm_values[3])
                / (norm_values[2] - norm_values[3])
            )
            + 10.0
        )
        x = torch.where(img_tensor == 0.0, img_tensor, final_tensor)
        return x

    elif normalization == "max_min":
        # img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))
        img_tensor = img_tensor - img_tensor.min()
        img_tensor = img_tensor / img_tensor.max()

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor


class MedicalImage3DDataset(Dataset):
    """DHCP-r2 dataset."""

    def __init__(
        self,
        mode,
        root_dir,
        modality,
        normalization="full_volume_mean",
        df_root_path=".",
        num_classes=10,
        labelled_if_train=False,
        seed=0,
        labelled_samples=0,
        val_samples=0,
        dataset="DHCP",
    ):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.dataset = dataset
        self.imgs = make_3Ddataset(
            root_dir,
            df_root_path,
            mode,
            modality,
            num_classes,
            labelled_if_train=labelled_if_train,
            seed=seed,
            labelled_samples=labelled_samples,
            val_samples=val_samples,
            dataset=dataset,
        )

    def transform_volume(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, -1)

        x = torch.from_numpy(x.transpose((-1, 0, 1, 2)))
        return x

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data_path, mask_path, GT_path, patch, name = self.imgs[index]
        img = nib.load(data_path).get_fdata(dtype=np.float32)
        mask = nib.load(mask_path).get_fdata(dtype=np.float32)
        gt = nib.load(GT_path).get_fdata(dtype=np.float32)

        # W_s = patch['W_s']
        # W_e = patch['W_e']
        # H_s = patch['H_s']
        # H_e = patch['H_e']
        # D_s = patch['D_s']
        # D_e = patch['D_e']

        # img = img[W_s:W_e, H_s:H_e, D_s:D_e]
        # gt = gt[W_s:W_e, H_s:H_e, D_s:D_e]
        # mask = mask[W_s:W_e, H_s:H_e, D_s:D_e]

        # All modality and mask should have same transform ???
        img = self.transform_volume(img)

        # Normalization
        """
        MEAN, STD, MAX, MIN = 0., 1., 1., 0.
        MEAN, STD = img_FLAIR.mean(), img_FLAIR.std()
        MAX, MIN = img_FLAIR.max(), img_FLAIR.min()
        img_FLAIR = normalize_intensity(img_FLAIR, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))
        """
        img = normalize_intensity(img, normalization="mean")

        # mask = mask.unsqueeze(dim=1)
        # print(img_T1w.shape, mask.shape)
        # print([img_T1w.max(), img_T1w.min(), img_T1w.mean()])
        # print([np.max(mask), np.min(mask), np.mean(mask)])

        return [img, mask, gt, name]
