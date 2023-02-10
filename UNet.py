from Blocks import *
import torch.nn.init as init
import torch.nn.functional as F
import pdb
import math
from layers import *


class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class unetUp(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_dim, out_dim, act_fn)

        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetConv2(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(unetConv2, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv2 = conv_block(self.out_dim, self.out_dim, act_fn)

        """self.conv1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(),)
                                   
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_dim, kernel_size, stride, padding),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU(),)"""

    def forward(self, input):
        conv_1 = self.conv1(input)
        conv_2 = self.conv2(conv_1)

        return conv_2


class UNetD(nn.Module):
    def __init__(self, nin, nD):
        super(UNetD, self).__init__()
        self.convs = nn.Sequential(
            convBatch(nin, nD),
            convBatch(nD, nD),
            convBatch(nD * 1, nD * 2, stride=2),
            convBatch(nD * 2, nD * 2),
            convBatch(nD * 2, nD * 4, stride=2),
            convBatch(nD * 4, nD * 4),
            convBatch(nD * 4, nD * 8, stride=2),
            convBatch(nD * 8, nD * 8),
            convBatch(nD * 8, nD * 16, stride=2),
            convBatch(nD * 16, nD * 16),
            convBatch(nD * 16, nD * 16, stride=2),
            convBatch(nD * 16, nD * 16),
            convBatch(nD * 16, nD * 16, stride=2),
            convBatch(nD * 16, nD * 16),
        )
        self.final = nn.Sequential(
            convBatch(nD * 16, nD * 16),
            convBatch(nD * 16, nD * 16, kernel_size=5, padding=0),
            nn.Conv2d(nD * 16, 1, kernel_size=1),
        )

    def forward(self, input):
        x = self.convs(input)
        x = self.final(x)

        return F.sigmoid(x)


class UNetG(nn.Module):
    def __init__(self, nin, nG, nout):
        super(UNetG, self).__init__()
        self.conv0 = nn.Sequential(convBatch(nin, nG), convBatch(nG, nG))
        self.conv1 = nn.Sequential(
            convBatch(nG * 1, nG * 2, stride=2), convBatch(nG * 2, nG * 2)
        )
        self.conv2 = nn.Sequential(
            convBatch(nG * 2, nG * 4, stride=2), convBatch(nG * 4, nG * 4)
        )
        self.conv3 = nn.Sequential(
            convBatch(nG * 4, nG * 8, stride=2), convBatch(nG * 8, nG * 8)
        )
        self.bridge = nn.Sequential(
            convBatch(nG * 8, nG * 16, stride=2),
            residualConv(nG * 16, nG * 16),
            convBatch(nG * 16, nG * 16),
        )

        self.deconv0 = upSampleConv(nG * 16, nG * 16)
        self.conv4 = nn.Sequential(
            convBatch(nG * 24, nG * 8), convBatch(nG * 8, nG * 8)
        )
        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(
            convBatch(nG * 12, nG * 4), convBatch(nG * 4, nG * 4)
        )
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2), convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1), convBatch(nG * 1, nG * 1))

        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        bridge = self.bridge(x3)

        y = self.deconv0(bridge)
        y = self.deconv1(self.conv4(torch.cat((y, x3), dim=1)))
        y = self.deconv2(self.conv5(torch.cat((y, x2), dim=1)))
        y = self.deconv3(self.conv6(torch.cat((y, x1), dim=1)))
        y = self.conv7(torch.cat((y, x0), dim=1))

        # return F.softmax(self.final(y), dim=1)
        return self.final(y)


class UNetDisc(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(UNetDisc, self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2)
        act_fn_2 = nn.ReLU()
        img_Size = 320
        # Encoder
        # self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.down_1 = unetConv2(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        # self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.down_2 = unetConv2(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        # self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.down_3 = unetConv2(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        # self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.down_4 = unetConv2(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        featMapSize = int(img_Size / 16)  # After 4 pooling ops

        # Get output for classification loss
        self.classOut = classificationNet(
            self.out_dim * 16 * featMapSize * featMapSize
        )  # Same input size as bridge

        # Params initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                # init.xavier_uniform(m.weight.data)
                # init.xavier_uniform(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):

        # ############################# #
        # ~~~~~~ Encoding path ~~~~~~~  #

        ## ~~~~~~ First Layer ~~~~~~ ###
        down_1 = self.down_1(input)  # This will go as res in deconv path
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        bridgeflatten = bridge.view(
            bridge.shape[0], int(bridge.numel() / bridge.shape[0])
        )  # Flatten to (batchSize, numElemsPerSample)

        class_out = self.classOut(bridgeflatten)  # Classification output

        return F.sigmoid(class_out)


class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(UNet, self).__init__()
        img_Size = 256
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2)
        act_fn_2 = nn.ReLU()

        # Encoder
        # self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.down_1 = unetConv2(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        # self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.down_2 = unetConv2(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        # self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.down_3 = unetConv2(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        # self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.down_4 = unetConv2(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # featMapSize = img_Size/16 # After 4 pooling ops

        # Get output for classification loss
        # self.classOut = classificationNet(self.out_dim * 16 * featMapSize * featMapSize) # Same input size as bridge

        # Decoder
        self.is_deconv = True
        self.up_concat4 = unetUp(self.out_dim * 16, self.out_dim * 8, act_fn)
        self.up_concat3 = unetUp(self.out_dim * 8, self.out_dim * 4, act_fn)
        self.up_concat2 = unetUp(self.out_dim * 4, self.out_dim * 2, act_fn)
        self.up_concat1 = unetUp(self.out_dim * 2, self.out_dim, act_fn)

        """self.deconv_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_decod_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)"""

        # self.out = nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, 1)

        # self.out_2 = nn.Tanh()

        # Params initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                # init.xavier_uniform(m.weight.data)
                # init.xavier_uniform(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):

        # ############################# #
        # ~~~~~~ Encoding path ~~~~~~~  #

        ## ~~~~~~ First Layer ~~~~~~ ###
        down_1 = self.down_1(input)  # This will go as res in deconv path
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        # bridgeflatten = bridge.view(bridge.shape[0],bridge.numel()/bridge.shape[0]) # Flatten to (batchSize, numElemsPerSample)

        # class_out = self.classOut(bridgeflatten) # Classification output

        # ############################# #
        # ~~~~~~ Decoding path ~~~~~~~  #
        """deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)  # Residual connection
        
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)  # Residual connection
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)  # Residual connection
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)  # Residual connection
        up_4 = self.up_4(skip_4)
        """
        up4 = self.up_concat4(down_4, bridge)
        up3 = self.up_concat3(down_3, up4)
        up2 = self.up_concat2(down_2, up3)
        up1 = self.up_concat1(down_1, up2)

        # Last output
        # out = self.out(up_4)
        out = self.out(up1)

        return F.softmax(out, dim=1)

    # To get the center of features maps for the dense connections
    def cropLayer(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        _lim = (layer_width - target_size) // 2
        return layer[:, :, _lim : (_lim + target_size), _lim : (_lim + target_size)]


class UNetG_Dilated_Progressive(nn.Module):
    def __init__(self, nin, nG, nout):
        super(UNetG_Dilated_Progressive, self).__init__()
        print("*" * 50)
        print("--------- Creating Dilated Progressive UNet network... ---")
        print("*" * 50)
        self.conv0 = nn.Sequential(
            convBatch(nin, nG, stride=1, dilation=1, padding=1),
            convBatch(nG, nG, stride=1, dilation=2, padding=2),
            convBatch(nG, nG, stride=1, dilation=4, padding=4),
        )

        self.conv1 = nn.Sequential(
            convBatch(nG * 1, nG * 2, stride=2, dilation=1, padding=1),
            convBatch(nG * 2, nG * 2, stride=1, dilation=2, padding=2),
            convBatch(nG * 2, nG * 2, stride=1, dilation=4, padding=4),
        )

        self.conv2 = nn.Sequential(
            convBatch(nG * 2, nG * 4, stride=2, dilation=1, padding=1),
            convBatch(nG * 4, nG * 4, stride=1, dilation=2, padding=2),
            convBatch(nG * 4, nG * 4, stride=1, dilation=4, padding=4),
        )

        self.conv3 = nn.Sequential(
            convBatch(nG * 4, nG * 8, stride=2, dilation=1, padding=1),
            convBatch(nG * 8, nG * 8, stride=1, dilation=2, padding=2),
            convBatch(nG * 8, nG * 8, stride=1, dilation=4, padding=4),
        )

        self.bridge = nn.Sequential(
            convBatch(nG * 8, nG * 16, stride=2, dilation=1, padding=1),
            residualConv(nG * 16, nG * 16),
            convBatch(nG * 16, nG * 16),
        )

        self.deconv0 = upSampleConv(nG * 16, nG * 16)
        self.conv4 = nn.Sequential(
            convBatch(nG * 24, nG * 8), convBatch(nG * 8, nG * 8)
        )
        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(
            convBatch(nG * 12, nG * 4), convBatch(nG * 4, nG * 4)
        )
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2), convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1), convBatch(nG * 1, nG * 1))

        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        bridge = self.bridge(x3)
        y = self.deconv0(bridge)

        y = self.deconv1(self.conv4(torch.cat((y, x3), dim=1)))
        y = self.deconv2(self.conv5(torch.cat((y, x2), dim=1)))
        y = self.deconv3(self.conv6(torch.cat((y, x1), dim=1)))
        y = self.conv7(torch.cat((y, x0), dim=1))

        return self.final(y)


if __name__ == "__main__":
    G = UNetG(1, 16, 4).cuda()
    from torch.autograd import Variable

    x = Variable(torch.randn(10, 1, 320, 320)).cuda()
    print(G(x).size())

    D = UNetD(4, 16).cuda()
    print(D(G(x)).size())
