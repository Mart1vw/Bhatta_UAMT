import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# 3D Unet
###############################################################################


class Unet3D(nn.Module):
    def __init__(
        self,
        n_layers=2,  # Number of Layers in each Step in Unet: Ex. 2
        n_scales=3,  # Number of scales in Unet: Ex. 3
        init_ker=4,  # Number of Initial Kernals in the Architecture: Ex. 8
        inp_chan=3,  # number of input channels ex. 4 for BraTS
        out_chan=1,  # number of output channels ex. 4 for BraTS
        dropout_rate=0.05,  # dropout rate: 0 < dr < 0.5
        norm="none",  # Normalization Type: BatchNorm("bn"), InstanceNorm("in"), LayerNorm("ln"), GroupNorm("gn"), or "none"
        activ="relu",  # Activation Type: "relu", LeakyReLU("lrelu"), "prelu", "selu", "tanh", or "none"
        pad_type="zero",  # Padding Type: "zero" or "replicate"
        ds_type="max",  # DownSample Type: AveragePooling("avg") or MaxPooling("max")
        us_type="up",  # UpSample Type: TransposedConv ("transpose") or UpSampling("upsample")
        merge_type="sum",  # Merge Type: Summation("sum"), Multiplication("mul") or Concatination("concate")
        return_layer=False,
    ):  # Return Layer : Bool used to return (or not) the feature map used to compute feature uncertaintainty
        super(Unet3D, self).__init__()

        self.n_layers = n_layers
        self.n_scales = n_scales
        self.init_ker = init_ker
        self.inp_chan = inp_chan
        self.out_chan = out_chan
        self.dropout_rate = dropout_rate
        self.norm = norm
        self.activ = activ
        self.pad_type = pad_type
        self.ds_type = ds_type
        self.us_type = us_type
        self.merge_type = merge_type
        self.return_layer = return_layer

        if self.ds_type == "avg":
            self.downsample = nn.AvgPool3d(2, stride=2, padding=0)
        elif self.ds_type == "max":
            self.downsample = nn.MaxPool3d(2, stride=2, padding=0)
        else:
            assert 0, "Unsupported pooling type: {}".format(ds_type)

        self.last_layer = nn.Conv3d(
            self.init_ker, self.out_chan, 1, 1, bias=True
        )  # to map to desired output channel size

        self.enc = self.encoder()
        self.dec = self.decoder()

    def encoder(self):

        inp_dim = self.inp_chan
        out_dim = self.init_ker

        model = nn.ModuleList()

        for j in range(self.n_scales):

            model.append(
                convBlocks(
                    inp_kern=inp_dim,
                    out_kern=out_dim,
                    kern_size=3,
                    norm=self.norm,
                    activ=self.activ,
                    pad_type=self.pad_type,
                    dropout_rate=self.dropout_rate,
                    n_layers=self.n_layers,
                )
            )

            inp_dim = out_dim
            out_dim *= 2

        return model

    def decoder(self):

        inp_dim = self.init_ker * (2 ** (self.n_scales - 1))
        out_dim = self.init_ker * (2 ** (self.n_scales - 2))

        model = nn.ModuleList()

        for _ in range(self.n_scales - 1):

            model.append(
                convTransBlocks(
                    inp_kern=inp_dim,
                    out_kern=out_dim,
                    kern_size=3,
                    norm=self.norm,
                    activ=self.activ,
                    pad_type=self.pad_type,
                    dropout_rate=self.dropout_rate,
                    n_layers=self.n_layers,
                    us_type=self.us_type,
                    merge_type=self.merge_type,
                )
            )

            inp_dim //= 2
            out_dim //= 2

        return model

    def forward(self, x, return_layer=False):

        enc_output = []

        for i in range(self.n_scales):
            x = self.enc[i](x)  # take each scale encoder and pass input (x) through it
            enc_output += [x]  # store output for decoder part
            if i < self.n_scales - 1:  # if not last scale than apply downsampling
                x = self.downsample(x)

        for i in range(self.n_scales - 1):
            x = self.dec[i](
                x, enc_output[abs(i - self.n_scales + 2)]
            )  # pass through deocoder scale with its corresponding encoder output

        output = self.last_layer(x)  # pass through last layer

        if self.return_layer:
            # print('that seems to be working :)')
            layer = enc_output[abs(1 - self.n_scales + 2)]
            # print('OUTPUT SHAPE :', layer.shape)
            return output, layer
        return output  # return


#####################################################################################
## Unet Blocks
######################################################################################


class convBlocks(nn.Module):
    def __init__(
        self,
        inp_kern=4,
        out_kern=1,
        kern_size=3,
        norm="bn",
        activ="relu",
        pad_type="zero",
        dropout_rate=0,
        n_layers=2,
    ):

        super(convBlocks, self).__init__()

        self.model = []

        self.model += [
            conv3dBlock(
                inp_kern,
                out_kern,
                kern_size,
                1,
                1,
                norm=norm,
                activation=activ,
                pad_type=pad_type,
                dropout_rate=0,
            )
        ]
        for _ in range(n_layers - 2):
            self.model += [
                conv3dBlock(
                    out_kern,
                    out_kern,
                    kern_size,
                    1,
                    1,
                    norm="none",
                    activation=activ,
                    pad_type=pad_type,
                    dropout_rate=0,
                )
            ]
        self.model += [
            conv3dBlock(
                out_kern,
                out_kern,
                kern_size,
                1,
                1,
                norm="none",
                activation=activ,
                pad_type=pad_type,
                dropout_rate=dropout_rate,
            )
        ]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class convTransBlocks(nn.Module):
    def __init__(
        self,
        inp_kern=4,
        out_kern=1,
        kern_size=3,
        norm="bn",
        activ="relu",
        pad_type="zero",
        dropout_rate=0,
        n_layers=2,
        us_type="transpose",
        merge_type="sum",
    ):
        super(convTransBlocks, self).__init__()

        self.merge_type = merge_type

        if us_type == "transpose":
            self.upsamp = convTranspose3dBlock(
                inp_kern,
                out_kern,
                3,
                2,
                1,
                1,
                norm="none",
                activation=activ,
                dropout_rate=dropout_rate,
            )
        else:
            m = []
            m += [nn.Upsample(scale_factor=2, mode="nearest")]
            m += [
                conv3dBlock(
                    inp_kern, out_kern, 3, 1, 1, activation=activ, pad_type=pad_type
                )
            ]
            self.upsamp = nn.Sequential(*m)

        if self.merge_type == "sum" or self.merge_type == "mul":
            in_kern = out_kern
        else:
            in_kern = inp_kern

        self.model = []

        self.model += [
            conv3dBlock(
                in_kern,
                out_kern,
                kern_size,
                1,
                1,
                norm=norm,
                activation=activ,
                pad_type=pad_type,
                dropout_rate=0,
            )
        ]
        for _ in range(n_layers - 2):
            self.model += [
                conv3dBlock(
                    out_kern,
                    out_kern,
                    kern_size,
                    1,
                    1,
                    norm="none",
                    activation=activ,
                    pad_type=pad_type,
                    dropout_rate=0,
                )
            ]
        self.model += [
            conv3dBlock(
                out_kern,
                out_kern,
                kern_size,
                1,
                1,
                norm="none",
                activation=activ,
                pad_type=pad_type,
                dropout_rate=0,
            )
        ]

        self.model = nn.Sequential(*self.model)

    def forward(self, x1, x2):
        x1 = self.upsamp(x1)
        if self.merge_type == "sum":
            x = x1 + x2
        elif self.merge_type == "mul":
            x = x1 * x2
        else:
            x = torch.cat((x1, x2), 1)
        return self.model(x)


#######################################################################################
# Basic Blocks
#######################################################################################


class conv3dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
        dropout_rate=0,
    ):

        super(conv3dBlock, self).__init__()

        self.use_bias = True
        self.dropout_rate = dropout_rate

        # initialize padding
        if pad_type == "replicate":
            self.pad = nn.ReplicationPad3d(padding)
        elif pad_type == "zero":
            self.pad = nn.ConstantPad3d(padding, 0)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = input_dim
        if norm == "bn":
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == "gn":
            self.norm = nn.GroupNorm(norm_dim // 2, norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if dropout_rate > 0:
            self.dp = nn.Dropout3d(dropout_rate)
        else:
            self.dp = None

        # initialize convolution
        self.conv = nn.Conv3d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=self.use_bias,
        )

    def forward(self, x):

        if self.norm:
            x = self.norm(x)
        x = self.conv(self.pad(x))
        if self.activation:
            x = self.activation(x)
        if self.dp:
            x = self.dp(x)

        return x


class convTranspose3dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
        norm="none",
        activation="relu",
        dropout_rate=0,
    ):

        super(convTranspose3dBlock, self).__init__()

        self.use_bias = True
        self.dropout_rate = dropout_rate

        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == "gn":
            self.norm = nn.GroupNorm(norm_dim // 2, norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if dropout_rate > 0:
            self.dp = nn.Dropout3d(dropout_rate)
        else:
            self.dp = None

        # initialize convolution
        self.convtp = nn.ConvTranspose3d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=self.use_bias,
        )

    def forward(self, x):

        x = self.convtp(x)
        if self.activation:
            x = self.activation(x)
        if self.norm:
            x = self.norm(x)
        if self.dp:
            x = self.dp(x)

        return x
