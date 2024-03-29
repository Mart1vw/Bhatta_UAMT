from os import device_encoding
from numpy import dtype
import torch
import torch.nn.functional as F
from torch import device, nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from time import time
from math import log


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C


def total_bhatta_tensor(tensor, bins=5, eps=10**-6, class_2=10):
    total = 0
    for number in range(2, class_2 + 1):

        total += adaptative_bhatta_tensor(tensor, bins=bins, eps=eps, class_2=number)

    return total


def adaptative_bhatta_tensor(tensor, bins=5, eps=10**-6, class_2=2):

    T = tensor.shape[0]
    # device = tensor.device

    mean_tensor = tensor.mean(dim=0, keepdim=True)  # (1, 1, 10, 64**3)

    sorted_tensor = mean_tensor.topk(dim=2, k=class_2)[1]
    # print(sorted_tensor.shape)                  # (1, 1, 10, 64**3)

    top_1_class = sorted_tensor.narrow(dim=2, start=0, length=1)  # (1, 1, 1, 64**3)
    top_2_class = sorted_tensor.narrow(dim=2, start=-1, length=1)  # (1, 1, 1, 64**3)

    result_1 = tensor.gather(
        dim=2, index=top_1_class.expand(T, -1, -1, -1, -1, -1)
    )  # (32, 1, 1, 64**3)
    result_2 = tensor.gather(
        dim=2, index=top_2_class.expand(T, -1, -1, -1, -1, -1)
    )  # (32, 1, 1, 64**3)

    bin_result_1 = (result_1 * bins).to(dtype=torch.uint8)
    bin_result_2 = (result_2 * bins).to(dtype=torch.uint8)

    H1, H2 = [], []
    for i in range(bins):
        H1.append((bin_result_1 == i).sum(dim=0))
        H2.append((bin_result_2 == i).sum(dim=0))

    H1 = torch.stack(H1, dim=0).to(dtype=tensor.dtype)
    H2 = torch.stack(H2, dim=0).to(dtype=tensor.dtype)

    H1.sqrt_().div_(T)
    H2.sqrt_()

    bhatta = (H1 * H2).sum(dim=0)

    return bhatta


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert (
        input.size() == target.size()
    ), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, "ignore_index cannot be None"
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert (
            target.size(1) > 1
        ), "Target tensor has a singleton channel dimension, cannot remove channel"

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization="sigmoid"):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer("weight", weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ["sigmoid", "softmax", "none"]
        if normalization == "sigmoid":
            self.normalization = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1.0 - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization="softmax"):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf."""

    def __init__(self, normalization="sigmoid", epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert (
            input.size() == target.size()
        ), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(
            input, target
        )


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf"""

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(
            input, target, weight=weight, ignore_index=self.ignore_index
        )

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1.0 - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(
            target, C=input.size()[1], ignore_index=self.ignore_index
        )
        # expand weights
        weights = weights.unsqueeze(1)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert "loss" in config, "Could not find loss function configuration"
    loss_config = config["loss"]
    name = loss_config.pop("name")

    ignore_index = loss_config.pop("ignore_index", None)
    skip_last_target = loss_config.pop("skip_last_target", False)
    weight = loss_config.pop("weight", None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config["device"])

    pos_weight = loss_config.pop("pos_weight", None)
    if pos_weight is not None:
        # convert to cuda tensor if necessary
        pos_weight = torch.tensor(pos_weight).to(config["device"])

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (
        ignore_index is None or name in ["CrossEntropyLoss", "WeightedCrossEntropyLoss"]
    ):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(
            loss, loss_config.get("squeeze_channel", False)
        )

    return loss


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction="none")
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


#######################################################################################################################


def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == "BCEDiceLoss":
        alpha = loss_config.get("alphs", 1.0)
        beta = loss_config.get("beta", 1.0)
        return BCEDiceLoss(alpha, beta)
    elif name == "CrossEntropyLoss":
        if ignore_index is None:
            ignore_index = (
                -100
            )  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == "WeightedCrossEntropyLoss":
        if ignore_index is None:
            ignore_index = (
                -100
            )  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == "PixelWiseCrossEntropyLoss":
        return PixelWiseCrossEntropyLoss(
            class_weights=weight, ignore_index=ignore_index
        )
    elif name == "GeneralizedDiceLoss":
        normalization = loss_config.get("normalization", "sigmoid")
        return GeneralizedDiceLoss(normalization=normalization)
    elif name == "DiceLoss":
        normalization = loss_config.get("normalization", "sigmoid")
        return DiceLoss(weight=weight, normalization=normalization)
    elif name == "MSELoss":
        return MSELoss()
    elif name == "SmoothL1Loss":
        return SmoothL1Loss()
    elif name == "L1Loss":
        return L1Loss()
    elif name == "WeightedSmoothL1Loss":
        return WeightedSmoothL1Loss(
            threshold=loss_config["threshold"],
            initial_weight=loss_config["initial_weight"],
            apply_below_threshold=loss_config.get("apply_below_threshold", True),
        )
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")


def bhatta_tensor(tensor, bin=5, eps=10**-6):
    T = tensor.shape[0]
    tensor = tensor.clamp(min=0, max=1 - eps)
    tensor = (
        torch.mul(tensor, bin)
    ).int()  # on passe les valeurs du tenseur de [0, 1] à [|0, bin|]
    H1 = torch.zeros([bin] + list(tensor.shape)[1:], device="cuda")  # l'histogramme
    for i in range(bin):
        bin_tensor = tensor == i
        bin_tensor = bin_tensor.sum(dim=0)
        H1[
            i
        ] = bin_tensor  # le nombre d'occurence de la bin i trouvées dans les T estimations de chaque pixel
    H1 = torch.sqrt(H1 * (1 / T))
    H2 = torch.flip(
        H1, [0]
    )  # on retourne pour calculer Bhatta, et on a plus qu'à sommer
    # print('HIST :', H1.shape)
    bhatta = (H1 * H2).sum(dim=0).cuda()
    # print('TENSOR', bhatta.shape)
    # print('tensor :', bhatta[0,0,0,0,0])
    # print(' ')

    return bhatta


@torch.no_grad()
def new_bhatta_tensor(tensor, bins=5, eps=10**-6):

    T = tensor.shape[0]
    # device = tensor.device

    mean_tensor = tensor.mean(dim=0, keepdim=True)  # (1, 1, 10, 64**3)

    sorted_tensor = mean_tensor.topk(dim=2, k=2)  # (1, 1, 10, 64**3)

    best_2_classes = sorted_tensor.indices  # (1, 1, 10, 64**3)

    top_1_class = best_2_classes.narrow(dim=2, start=0, length=1)  # (1, 1, 1, 64**3)
    top_2_class = best_2_classes.narrow(dim=2, start=1, length=1)  # (1, 1, 1, 64**3)

    result_1 = tensor.gather(
        dim=2, index=top_1_class.expand(T, -1, -1, -1, -1, -1)
    )  # (32, 1, 1, 64**3)
    result_2 = tensor.gather(
        dim=2, index=top_2_class.expand(T, -1, -1, -1, -1, -1)
    )  # (32, 1, 1, 64**3)

    bin_result_1 = (result_1 * bins).to(dtype=torch.uint8)
    bin_result_2 = (result_2 * bins).to(dtype=torch.uint8)

    H1, H2 = [], []
    for i in range(bins):
        H1.append((bin_result_1 == i).sum(dim=0))
        H2.append((bin_result_2 == i).sum(dim=0))

    H1 = torch.stack(H1, dim=0).to(dtype=tensor.dtype)
    H2 = torch.stack(H2, dim=0).to(dtype=tensor.dtype)

    H1.sqrt_().div_(T)
    H2.sqrt_()

    bhatta = (H1 * H2).sum(dim=0)

    return bhatta


def alpha_div(tensor, alpha, bins=5, eps=10**-6):

    T = tensor.shape[0]
    # device = tensor.device

    bins = T // 6

    mean_tensor = tensor.mean(dim=0, keepdim=True)  # (1, 1, 10, 64**3)

    sorted_tensor = mean_tensor.topk(dim=2, k=2)  # (1, 1, 10, 64**3)

    best_2_classes = sorted_tensor.indices  # (1, 1, 10, 64**3)

    top_1_class = best_2_classes.narrow(dim=2, start=0, length=1)  # (1, 1, 1, 64**3)
    top_2_class = best_2_classes.narrow(dim=2, start=1, length=1)  # (1, 1, 1, 64**3)

    result_1 = tensor.gather(
        dim=2, index=top_1_class.expand(T, -1, -1, -1, -1, -1)
    )  # (32, 1, 1, 64**3)
    result_2 = tensor.gather(
        dim=2, index=top_2_class.expand(T, -1, -1, -1, -1, -1)
    )  # (32, 1, 1, 64**3)

    bin_result_1 = (result_1 * bins).to(dtype=torch.uint8)
    bin_result_2 = (result_2 * bins).to(dtype=torch.uint8)

    H1, H2 = [], []
    for i in range(bins):
        H1.append((bin_result_1 == i).sum(dim=0))
        H2.append((bin_result_2 == i).sum(dim=0))

    H1 = torch.stack(H1, dim=0).to(dtype=tensor.dtype).div_(T)
    H2 = torch.stack(H2, dim=0).to(dtype=tensor.dtype).div_(T)
    print(" ")
    print("H1 bounds : ", H1.min(), H1.max())
    print(" ")

    alpha_d = torch.tensor(1.0 / (1 - alpha)) * (
        1 - (torch.pow(H1 + eps, (alpha)) * torch.pow(H2 + eps, (1 - alpha))).sum(dim=0)
    )

    print("alpha bounds", torch.max(alpha_d), torch.min(alpha_d))

    return alpha_d


def double_uncert(seg, guide, uncert, eps=1e-6, beta=1e-3):
    # print(seg.shape, guide.shape, uncert.shape)

    # print(seg[0, :, 20, 20, 32])
    # print(guide[0, :, 20, 20, 32])
    # print(uncert[0, :, 20:25, 20:25, 32])
    guide = (1 - uncert) * guide + uncert * seg
    loss = torch.log(guide + eps) * seg + beta * torch.log(1 - uncert + eps)
    # print("LOSS :", loss)
    # print('AAAAA', '\n')
    # print(torch.sum(loss, dim=(0, 1, 2, 3, 4)))
    # print(torch.log((1-uncert)).sum(dim=(0, 1, 2, 3 ,4)))
    # print(- ((torch.sum(loss, dim=(0, 1, 2, 3, 4))) + beta*torch.log((1-uncert)).sum(dim=(0, 1, 2, 3 ,4)))/ (seg.shape[-3]*seg.shape[-2]*seg.shape[-1]))
    return -(
        (torch.sum(loss, dim=(0, 1, 2, 3, 4)))
        + beta * torch.log((1 - uncert + eps)).sum(dim=(0, 1, 2, 3, 4))
    ) / (seg.shape[0] * seg.shape[-3] * seg.shape[-2] * seg.shape[-1])
