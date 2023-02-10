import os
from torch.utils.data import DataLoader
from progressBar import printProgressBar
import medicalDataLoader

from UNet3D import *
from utils import *

from losses import (
    DiceLoss,
    new_bhatta_tensor,
    softmax_mse_loss,
    softmax_kl_loss,
    alpha_div,
    double_uncert,
)
import time
import random
import torch
import numpy as np


from ramps import sigmoid_rampup, exp_rampup
from numpy import log


def runTraining(
    labelled_samples=3,
    val_samples=1,
    lr=0.001,
    epoch=100,
    N=3000,
    uncert="new_bhatta",
    consistency_threshold=0,
    seed=42,
    batch_size=6,
    batch_size_val=6,
    batch_size_unlabelled=6,
    ema_decay=0.99,
    consistency=1e5,
    consistency_type="mse",
    T=32,
    modality="T2",
    x_patch=64,
    y_patch=64,
    z_patch=64,
    num_classes=10,
    dataset="DHCP",
    method="UAMT",
    root_dir="../../../../home/ar94660/UNet3D_DHCP",
    main_path="Results-partial-T2/Statistics/",
    model_dir="Results-partial-T2/model/",
):
    """This function creates and trains the model from all the required parameters and paths.

    Args:
        labelled_samples (int): Number of labelled samples to use for training.
        val_samples (int): Number of labelled samples to use for validation.
        lr (float): Learning rate.
        epoch (int): Maximum number of epoch to go through.
        N (int): Number of patches N examined per unsupervised epoch. All patches are not examined by default because of long uncertainty computation time.
        uncert (string): Type of uncertainty used for the unsupervised learning. Choose from ["new_bhatta", "bhatta", None, "UAMT", "div-alpha" (replace alpha by value)].
        consistency_threshold (float): Threshold under which to avoid going through the unsupervised learning part of each epoch.
        seed (int): Seed for reproducibility.
        batch_size (int): Batch size for labelled training.
        batch_size_val (int): Batch size for validation.
        batch_size_unlabelled (int): Batch size for unlabelled training (best to take values <= batch_size).
        ema_decay (float): Speed of the teacher model's EMA decay.
        consistency (float): Consistency weight (higher ===>> more importance to unlabelled loss).
        consistency_type (string): Type of consistency to use when computing consistency_criterion. Can be "mse" or "kl".
        T (int): Number of times noised inference w/ dropout is computed on each patch. Take even numbers to avoid headaches.
        modality (string): Modality of imaging used. Is T2 for us.
        x_patch (int): Length of each patch along the x axis.
        y_patch (int): Length of each patch along the y axis.
        z_patch (int): Length of each patch along the z axis.
        num_classes (int): Total number of classes to segment to.
        dataset (str, optional): Dataset to be used. Defaults to "DHCP".
        method (str, optional): Paper from which the base model should be implemented. Can be "UAMT" or "DUAMT". Defaults to "UAMT".
        root_dir (str, optional): Path towards the images. Defaults to "../../../../home/ar94660/UNet3D_DHCP".
        mainPath (str, optional): Path to save statistics and others to.. Defaults to "Results-partial-T2/Statistics/".
        model_dir (str, optional): Path to save the model to. Defaults to "Results-partial-T2/model/".
    """

    modelName = f"{method}-{labelled_samples}_labels-{epoch}_epoch_" + str(uncert)
    print(f"Training model {modelName}")
    main_path = os.path.join(main_path, "modelName")

    P = (  # parameters
        labelled_samples,
        val_samples,
        lr,
        epoch,
        N,
        uncert,
        consistency_threshold,
        seed,
        batch_size,
        batch_size_val,
        batch_size_unlabelled,
        ema_decay,
        consistency,
        consistency_type,
        T,
        modality,
        x_patch,
        y_patch,
        z_patch,
        num_classes,
    )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_set = medicalDataLoader.MedicalImage3DDataset(
        "train",
        root_dir,
        modality,
        num_classes=num_classes,
        df_root_path=root_dir,
        labelled_if_train=True,
        seed=seed,
        labelled_samples=labelled_samples,
        dataset=dataset,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=min(batch_size, len(train_set)),
        num_workers=8,
        shuffle=True,
    )

    unlabelled_train_set = medicalDataLoader.MedicalImage3DDataset(
        "train",
        root_dir,
        modality,
        num_classes=num_classes,
        df_root_path=root_dir,
        labelled_if_train=False,
        seed=seed,
        labelled_samples=labelled_samples,
        dataset=dataset,
    )
    unlabelled_train_loader = DataLoader(
        unlabelled_train_set,
        batch_size=min(batch_size_unlabelled, len(unlabelled_train_set)),
        num_workers=8,
        shuffle=True,
    )

    val_set = medicalDataLoader.MedicalImage3DDataset(
        "val",
        root_dir,
        modality,
        num_classes=num_classes,
        df_root_path=root_dir,
        val_samples=val_samples,
        dataset=dataset,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(batch_size_val, len(val_set)),
        num_workers=8,
        shuffle=False,
    )

    print("TOTAL BATCHES:", len(val_loader), len(val_set))

    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")

    initial_kernels = 16
    ip_channel = 1

    # Load network
    netG = Unet3D(
        init_ker=initial_kernels,
        inp_chan=ip_channel,
        out_chan=num_classes,
        norm="in",
        activ="lrelu",
        n_scales=4,
    )
    netT = Unet3D(
        init_ker=initial_kernels,
        inp_chan=ip_channel,
        out_chan=num_classes,
        norm="in",
        activ="lrelu",
        n_scales=4,
        return_layer=(method == "DUAMT"),
    )

    for param in netT.parameters():
        param.detach_()

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = DiceLoss(normalization="none")

    if torch.cuda.is_available():
        netG.cuda()
        netT.cuda()
        netG = nn.DataParallel(netG)
        netT = nn.DataParallel(netT)
        softMax.cuda()
        CE_loss.cuda()
        Dice_loss.cuda()
        Dice_loss.cuda()

    if consistency_type == "mse":
        consistency_criterion = softmax_mse_loss
    elif consistency_type == "kl":
        consistency_criterion = softmax_kl_loss
    else:
        assert False, f"{consistency_type} is not implemented as a consistency type"

    optimizerG = torch.optim.Adam(
        netG.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False
    )
    BestDice, BestTotal, BestCE, BestEpoch = 1000, 1000, 1000, 0

    Losses_total = []
    Losses_CE = []
    Losses_Dice = []

    Losses_val_total = []
    Losses_val_CE = []
    Losses_val_Dice = []

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    print("-" * 40)
    for i in range(epoch):
        loss_total = []
        loss_Dice = []
        loss_CE = []
        if method == "UAMT":
            consistency_weight = sigmoid_rampup(i, epoch, consistency, N)
        else:
            consistency_weight = exp_rampup(i, epoch, consistency, N)
        labelled_batches = len(train_loader)
        unlabelled_batches = len(unlabelled_train_loader)
        val_batches = len(val_loader)

        start_time = time.time()

        for j, data in enumerate(train_loader):

            batch_num = i * (labelled_batches + unlabelled_batches) + j
            image, mask, labels, img_names = data[:4]

            if image.size(0) != batch_size:
                continue  # prevent batchnorm error for batch of size 1

            netG.train()
            optimizerG.zero_grad()

            MRI = to_var(image)
            Segmentation = to_var(labels)

            ################### Train ###################
            netG.zero_grad()

            # Logits
            segmentation_prediction = netG(MRI)

            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)
            CE_lossG = CE_loss(segmentation_prediction, Segmentation_class)

            # To softmax
            predClass_y = softMax(segmentation_prediction)

            # OneHot for Dice
            Segmentation_planes = getOneHotSegmentation(Segmentation, num_classes)
            Dice_lossG = Dice_loss(predClass_y, Segmentation_planes)

            lossG = CE_lossG + Dice_lossG

            if i / epoch >= consistency_threshold:
                noise = torch.clamp(torch.randn_like(MRI) * 0.1, -0.2, 0.2)
                ema_inputs = MRI + noise

                with torch.no_grad():
                    if method == "UAMT":
                        ema_output = netT(ema_inputs)
                    else:
                        ema_output, _ = netT(ema_inputs)

                volume_batch_r = MRI.repeat(2, 1, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2

                feature_maps = []
                preds = torch.zeros(
                    [stride * T, num_classes, x_patch, y_patch, z_patch],
                    device=ema_inputs.device,
                )

                for k in range(
                    T // 2
                ):  # This is the way this part is coded in the original UAMT paper, so it was replicated here.
                    ema_inputs = volume_batch_r + torch.clamp(
                        torch.randn_like(volume_batch_r, device=preds.device) * 0.1,
                        -0.2,
                        0.2,
                    )

                    with torch.no_grad():
                        if method == "UAMT":
                            preds[2 * stride * k : 2 * stride * (k + 1)] = netT(
                                ema_inputs
                            )
                        else:
                            (
                                preds[2 * stride * k : 2 * stride * (k + 1)],
                                feature_map,
                            ) = netT(ema_inputs)
                            feature_maps.append(feature_map[:stride])
                            feature_maps.append(feature_map[stride:])

                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, num_classes, x_patch, y_patch, z_patch)

                if uncert == "bhatta":
                    uncertainty = bhatta_tensor(preds)
                elif uncert == None or uncert == "None":
                    uncertainty = torch.zeros(preds.shape[1:])
                elif uncert == "UAMT":
                    preds = torch.mean(preds, dim=0)
                    uncertainty = (
                        -1.0
                        * torch.sum(
                            preds * torch.log(preds + 1e-6), dim=1, keepdim=True
                        )
                        / log(num_classes)
                    )
                elif uncert == "new_bhatta":
                    uncertainty = new_bhatta_tensor(preds)
                elif uncert[:3] == "div":
                    uncertainty = alpha_div(preds, float(uncert[4:]))
                else:
                    assert False, "Non-Existant Uncertainty Type"

                if method == "UAMT":
                    consistency_dist = consistency_criterion(
                        segmentation_prediction, ema_output
                    )
                    threshold = (0.75 + 0.25 * sigmoid_rampup(i, epoch)) * np.log(2)
                    mask = (uncertainty < threshold).float()
                    consistency_dist = torch.sum(mask * consistency_dist) / (
                        2 * torch.sum(mask) + 1e-16
                    )
                    consistency_loss = consistency_weight * consistency_dist

                else:
                    consistency_dist = double_uncert(
                        predClass_y, F.softmax(ema_output, dim=1), uncertainty
                    )
                    with torch.no_grad():
                        feature_maps = torch.stack(feature_maps)
                        dist_maps = torch.zeros(
                            (int(T * (T - 1) / 2),) + tuple(feature_maps.shape[1:]),
                            device=feature_maps.device,
                        )
                        depth = -1
                        for m1 in range(T):
                            for m2 in range(
                                m1 + 1, T
                            ):  # It is not clear how this quadratic cost could be avoided.
                                depth += 1
                                dist_maps[depth] = torch.sub(
                                    feature_maps[m1], feature_maps[m2]
                                ).abs_()
                        std_map = dist_maps.std(dim=0, unbiased=True)
                        std_map = std_map.mean(dim=(0, -1, -2, -3))
                        min_std = std_map.min()
                        feature_uncert = (std_map - min_std).sum(dim=0) / num_classes
                        seg_uncert = uncertainty.mean(dim=(0, 1, 2, 3, 4))
                    consistency_loss = (
                        -consistency_weight
                        * consistency_dist
                        / (feature_uncert + 1e-7)
                        * log(seg_uncert + 1e-7)
                    )

            else:
                consistency_loss = 0

            global_loss = lossG + consistency_loss

            global_loss.backward()
            optimizerG.step()
            update_ema_variables(netG, netT, batch_num, ema_decay=ema_decay)

            # This is a sanity check
            if method == "UAMT" and not (
                lossG.cpu().data.numpy() > 0 and lossG.cpu().data.numpy() < 10
            ):
                print("*************")
                print(img_names)
                print("*************")
                break

            loss_total.append(lossG.cpu().data.numpy())
            loss_CE.append(CE_lossG.cpu().data.numpy())
            loss_Dice.append(Dice_lossG.cpu().data.numpy())

            printProgressBar(
                j + 1,
                labelled_batches,
                prefix="[Labelled Training] Epoch: {} ".format(i),
                length=15,
                suffix=" loss_total: {:.4f},  loss_CE: {:.4g},  loss_Dice: {:.4f}, unsup_loss: {:.4f}".format(
                    lossG.data, CE_lossG.data, Dice_lossG.data, consistency_loss
                ),
            )

        if i / epoch >= consistency_threshold:
            for j, data in enumerate(unlabelled_train_loader):
                if j * batch_size_unlabelled > N:
                    break
                batch_num = (
                    i * (unlabelled_batches + labelled_batches) + j + labelled_batches
                )
                image, mask, _, img_names = data[:4]

                # prevent batchnorm error for batch of size 1
                if image.size(0) != batch_size:
                    continue

                netG.train()
                optimizerG.zero_grad()
                MRI = to_var(image)

                ################### Unsupervised Train ###################
                netG.zero_grad()

                # Logits
                segmentation_prediction = netG(MRI)
                predClass_y = softMax(segmentation_prediction)
                predClass_y.require_grad = True

                noise = torch.clamp(torch.randn_like(MRI) * 0.1, -0.2, 0.2)
                ema_inputs = to_var(MRI + noise)
                with torch.no_grad():
                    if method == "UAMT":
                        ema_output = netT(ema_inputs)
                    else:
                        ema_output, _ = netT(ema_inputs)

                feature_maps = []
                volume_batch_r = MRI.repeat(2, 1, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros(
                    [stride * T, num_classes, x_patch, y_patch, z_patch]
                ).cuda()
                for k in range(T // 2):
                    ema_inputs = volume_batch_r + torch.clamp(
                        torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2
                    )
                    with torch.no_grad():

                        if method == "UAMT":
                            preds[2 * stride * k : 2 * stride * (k + 1)] = netT(
                                ema_inputs
                            )
                        else:
                            (
                                preds[2 * stride * k : 2 * stride * (k + 1)],
                                feature_map,
                            ) = netT(ema_inputs)
                            feature_maps.append(feature_map[:stride])
                            feature_maps.append(feature_map[stride:])
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, num_classes, x_patch, y_patch, z_patch)

                if uncert == "bhatta":
                    uncertainty = bhatta_tensor(preds)
                elif uncert == "UAMT":
                    preds = torch.mean(preds, dim=0)  # (batch, 2, 112,112,80)
                    uncertainty = (
                        -1.0
                        * torch.sum(
                            preds * torch.log(preds + 1e-6), dim=1, keepdim=True
                        )
                        / log(num_classes)
                    )
                elif uncert == None or uncert == "None":
                    uncertainty = torch.zeros(preds.shape[1:], device="cuda")
                elif uncert == "new_bhatta":
                    uncertainty = new_bhatta_tensor(preds)
                elif uncert[:3] == "div":
                    uncertainty = alpha_div(preds, float(uncert[4:]))
                else:
                    assert False, "Non-Existant Uncertainty Type"

                if method == "UAMT":
                    consistency_dist = consistency_criterion(
                        segmentation_prediction, ema_output
                    )
                    threshold = (0.75 + 0.25 * sigmoid_rampup(i, epoch)) * np.log(2)
                    mask = (uncertainty < threshold).float()
                    consistency_dist = torch.sum(mask * consistency_dist) / (
                        2 * torch.sum(mask) + 1e-16
                    )
                    consistency_loss = consistency_weight * consistency_dist
                else:
                    consistency_dist = double_uncert(
                        predClass_y, F.softmax(ema_output, dim=1), uncertainty
                    )

                    with torch.no_grad():
                        feature_maps = torch.stack(feature_maps)
                        dist_maps = torch.zeros(
                            (int(T * (T - 1) / 2),) + tuple(feature_maps.shape[1:]),
                            device=feature_maps.device,
                        )
                        depth = -1
                        for m1 in range(T):
                            for m2 in range(m1 + 1, T):
                                depth += 1
                                dist_maps[depth] = torch.sub(
                                    feature_maps[m1], feature_maps[m2]
                                ).abs_()

                        std_map = dist_maps.std(dim=0, unbiased=True)
                        std_map = std_map.mean(dim=(0, -1, -2, -3))
                        min_std = std_map.min()
                        feature_uncert = (std_map - min_std).sum(dim=0)
                        seg_uncert = uncertainty.mean(dim=(0, 1, 2, 3, 4))

                    consistency_loss = (
                        -consistency_weight
                        * consistency_dist
                        / (feature_uncert + 1e-7)
                        * log(seg_uncert + 1e-7)
                    )
                global_loss = consistency_loss

                global_loss.backward()
                optimizerG.step()
                update_ema_variables(netG, netT, batch_num, ema_decay=ema_decay)
                printProgressBar(
                    (j + 1) * batch_size_unlabelled,
                    N,
                    prefix="[Unlabelled Training] Epoch: {} ".format(i),
                    length=15,
                    suffix=" loss_unsup: {:.4f}".format(global_loss),
                )

        directory = mainPath
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, "train-Losses.npy"), Losses_total)
        np.save(os.path.join(directory, "train-Losses_CE.npy"), Losses_CE)
        np.save(os.path.join(directory, "train-Losses_Dice.npy"), Losses_Dice)

        loss_val_total = []
        loss_val_Dice = []
        loss_val_CE = []
        for j, data in enumerate(val_loader):
            image, labels = data[0], data[2]
            if image.shape[0] != batch_size_val:
                print("SKETCHY PATCH, IGNORING IT")  # sanity check
                continue

            netG.eval()
            optimizerG.zero_grad()
            MRI = to_var(image)
            Segmentation = to_var(labels)

            ################### Validation ###################
            netG.zero_grad()
            segmentation_prediction = netG(MRI)
            predClass_y = softMax(segmentation_prediction)
            Segmentation_class = getTargetSegmentation(Segmentation)
            CE_lossG = CE_loss(segmentation_prediction, Segmentation_class)
            Segmentation_planes = getOneHotSegmentation(Segmentation, num_classes)
            Dice_lossG = Dice_loss(predClass_y, Segmentation_planes)
            lossG = CE_lossG + Dice_lossG

            # Save for plots
            loss_val_total.append(lossG.cpu().data.numpy())
            loss_val_CE.append(CE_lossG.cpu().data.numpy())
            loss_val_Dice.append(Dice_lossG.cpu().data.numpy())

            printProgressBar(
                j + 1,
                val_batches,
                prefix="[validation] Epoch: {} ".format(i),
                length=15,
                suffix=" loss_val_total: {:.4f},  loss_val_CE: {:.4f},  loss_val_Dice: {:.4f}".format(
                    lossG.data, CE_lossG.data, Dice_lossG.data
                ),
            )

        Losses_val_total.append(np.mean(loss_val_total))
        Losses_val_CE.append(np.mean(loss_val_CE))
        Losses_val_Dice.append(np.mean(loss_val_Dice))

        np.save(os.path.join(directory, "val-Losses.npy"), Losses_val_total)
        np.save(os.path.join(directory, "val-Losses_CE.npy"), Losses_val_CE)
        np.save(os.path.join(directory, "val-Losses_Dice.npy"), Losses_val_Dice)

        printProgressBar(
            labelled_batches,
            unlabelled_batches,
            done="[Training] Epoch: {}, Loss_total: {:.4f}, Loss_CE: {:.4f}, Loss_Dice: {:.4f}".format(
                i, np.mean(loss_val_total), np.mean(loss_val_CE), np.mean(loss_val_Dice)
            ),
        )

        CurrentDice = np.mean(loss_val_Dice)

        if CurrentDice < BestDice:
            BestEpoch = i
            BestCE = np.mean(loss_val_CE)
            BestDice = CurrentDice
            BestTotal = BestDice + BestCE

            print(
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            )
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            print("path and repo :")
            print(model_dir)
            print(os.getcwd())
            print(
                "SAVING MODEL IN ",
                os.path.join(model_dir, "Best_" + modelName + str(P) + ".pkl"),
            )
            torch.save(
                netG, os.path.join(model_dir, "Best_" + modelName + str(P) + ".pkl")
            )

        print(
            "###                                                                                  ###"
        )
        print(
            "### Best Dice Loss(mean): {:.4f} at epoch {} with (Dice loss): {:.4f}  (Total Loss): {:.4f} (CE Loss): {:.4f} ###".format(
                BestDice, BestEpoch, BestDice, BestTotal, BestCE
            )
        )
        print(
            "###                                                                                  ###"
        )
        print(
            "Time spent for entire epoch:  {:.4f}".format(time.time() - start_time),
            "with parameters :",
            P,
        )
        print(" ")
        print(" ")

        if i % 15 == 14:
            for param_group in optimizerG.param_groups:
                param_group["lr"] = lr / 2
    print(f"Finished training for {modelName} with parameters \n {P}")


def update_ema_variables(model, ema_model, global_step, ema_decay=0.99):
    """Update all the weights in the teacher model to follow the student's weights with an EMA.

    Args:
        model (_type_): Network to be used as reference ofr the EMA.
        ema_model (_type_): Model to be updated.
        global_step (int): Number of patches treated this far.
        ema_decay (float, optional): Speed of the EMA decay. Defaults to 0.99.
    """
    ema_decay = min(1 - 1 / (global_step + 1), ema_decay)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(ema_decay).add_(1 - ema_decay, param.data)


if __name__ == "__main__":
    runTraining()
