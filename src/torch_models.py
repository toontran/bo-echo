'''
Joshua Stough

Torch models, starting with a UNet-like construct mostly ripped 
from an Nvidia DLI course on VNet for prostate segmentation. 
'''

# torch
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch import nn
from torch.autograd import Variable
from torch.nn import Module, Conv2d, Parameter

import os


class GroupNorm2D(Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm2D, self).__init__()
        self.weight = Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
    

class ResidualConvBlock(Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none', expand_chan=False):
        super(ResidualConvBlock, self).__init__()

        self.expand_chan = expand_chan
        if self.expand_chan:
            ops = []

            ops.append(nn.Conv2d(n_filters_in, n_filters_out, 1))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm2D(n_filters_out))

            ops.append(nn.ReLU(inplace=True))

            self.conv_expan = nn.Sequential(*ops)

        ops = []
        for i in range(n_stages):
            if normalization != 'none':
                ops.append(nn.Conv2d(n_filters_in, n_filters_out, 3, padding=1))
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm2d(n_filters_out))
                if normalization == 'groupnorm':
                    ops.append(GroupNorm2D(n_filters_out))
            else:
                ops.append(nn.Conv2d(n_filters_in, n_filters_out, 3, padding=1))

            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    # I think this is adding and not concatenating...
    def forward(self, x):
        if self.expand_chan:
            x = self.conv(x) + self.conv_expan(x)
        else:
            x = (self.conv(x) + x)

        return x
    
# Now the down and upsampling layers

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm2D(n_filters_out))
        else:
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            if normalization == 'groupnorm':
                ops.append(GroupNorm2D(n_filters_out))
        else:
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    

'''
UNetLike is a UNet-like 2D pixel-wise (semantic) segmentation pipeline with residual (+, not concat) 
connections. 
'''    
class UNetLike(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=16, normalization='none'):
        super(UNetLike, self).__init__()

        if n_channels > 1:
            self.block_one = ResidualConvBlock(1, n_channels, n_filters, normalization=normalization, expand_chan=True)
        else:
            self.block_one = ResidualConvBlock(1, n_channels, n_filters, normalization=normalization)

        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv2d(n_filters, n_classes, 1, padding=0)
        
        self.final_sig = torch.nn.Sigmoid()
        
        # Cross entroy loss obviates the need to use this
        # See https://discuss.pytorch.org/t/do-i-need-to-use-softmax-before-nn-crossentropyloss/16739
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
#         self.softmax = nn.Softmax2d(dim=0)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4 # Here's where we could try concat instead.

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)

        f_out = self.out_conv(x9)
        
        out = self.final_sig(f_out)

        return out
    
    
'''
##################################################
Codes for shape prior-learning autoencoder.
##################################################
'''

class ConvBlock(Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out):
        super(ConvBlock, self).__init__()
        ops = []
        
        for i in range(n_stages):
            # Need padding to maintain padding with the kernel size
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, 
                                 kernel_size=3, padding=1))
            ops.append(nn.BatchNorm2d(n_filters_out))
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class DownConv(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, norm=False):
        super(DownConv, self).__init__()
        ops = []
        
        ops.append(nn.Conv2d(n_filters_in, n_filters_out, 
                             kernel_size=stride, stride=stride, padding=0))
        
        if norm:
            ops.append(nn.BatchNorm2d(n_filters_out))
        
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class UpConv(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, norm=False):
        super(UpConv, self).__init__()
        ops = []
        
        ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, 
                                      kernel_size=stride, stride=stride, padding=0))
        
        if norm:
            ops.append(nn.BatchNorm2d(n_filters_out))
        
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
    
    def forward(self, x):
        x = self.conv(x)
        return x

    
    
'''
## Autoencoder class

In the end, I want to be able to use the below AutoEncoder class to both encode and retrieve the encodings, and to push my own custom codes through the decoder section. The first problem can be solve by registering a forward hook with one of the modules in the class (self.fc1, the output of which is the encoding). The second problem I believe will be solved with conditionals inside the forward function, which should be fine according to documentation. Actually with lots of parameters it seems like forward could sometimes do encode, sometimes decode, and other times go all the way through without the difficulty of forward hooks. but anyway...

Wait, if we're the ones calling forward (when we call net(inputs) for example), then I don't need an optional_codes parameter or something, really just need an argument that tells forward whether the input should be encoded, decoded or run through the whole thing.

[This post](https://discuss.pytorch.org/t/how-to-use-condition-flow/644/5) says you can add conditionals into the forward method, while [this post](https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462/2) says you can send multiple arguments to the forward method. 
'''
    
class AutoEncoder(nn.Module):
    def __init__(self, encoding_dim=64, nfilters=16):
        super(AutoEncoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.nfilters = nfilters
        
        self.ae_block_one = ConvBlock(n_stages=1, n_filters_in=4, n_filters_out=nfilters) # nfilters x 256 x 256
        self.ae_down_one = DownConv(nfilters, 2*nfilters) # 2*nfilters x 128 x 128
        
        self.ae_block_two = ConvBlock(2, 2*nfilters, 2*nfilters)
        self.ae_down_two = DownConv(2*nfilters, 4*nfilters) # 4*nfilters x 64 x 64
        
        self.ae_block_three = ConvBlock(2, 4*nfilters, 4*nfilters)
        self.ae_down_three = DownConv(4*nfilters, 8*nfilters) # 8*nfilters x 32 x 32
        
        # Serious compression here, where we go to nfilters and downsample.
        self.ae_down_four = DownConv(8*nfilters, nfilters) # nfilters x 16 x 16
        
        
        # The encoded/latest space.
        self.ae_fc1 = nn.Linear(nfilters*16*16, self.encoding_dim)
        
        # Initial upsample
        self.ae_fc2 = nn.Linear(self.encoding_dim, 16**2) # 16 x 16 for the up direction.
        
        self.ae_up_four = UpConv(1, 8*nfilters) # 8*nfilters x 32 x 32
        self.ae_block_four = ConvBlock(2, 8*nfilters, 8*nfilters)
        
        self.ae_up_three = UpConv(8*nfilters, 4*nfilters) # 4*nfilters x 64 x 64
        self.ae_block_five = ConvBlock(2, 4*nfilters, 4*nfilters)
        
        self.ae_up_two = UpConv(4*nfilters, 2*nfilters) # 2*nfilters x 128 x 128
        self.ae_block_six = ConvBlock(2, 2*nfilters, 2*nfilters)
        
        self.ae_up_one = UpConv(2*nfilters, nfilters) # nfilters x 256 x 256
        
        # Last layer, without batch norm or ReLU
        self.ae_out_conv = nn.Conv2d(nfilters, out_channels=4, kernel_size=1, padding=0)
        
        self.ae_final_sig = torch.nn.Sigmoid()
        
        
    def forward(self, x, interpret_x=0):
        '''
        interpret_x:
        -1: encode only
        0 : push x through the whole thing
        1 : decode only
        '''
        if (interpret_x != 1):
            # whether it's encode_only or the whole deal, do this encoding part.
            
            x1 = self.ae_block_one(x)
            x1_dw = self.ae_down_one(x1)

            x2 = self.ae_block_two(x1_dw)
            x2_dw = self.ae_down_two(x2)
            # print('size after down_two: {}'.format(x2_dw.size()))
            
            x3 = self.ae_block_three(x2_dw)
            x3_dw = self.ae_down_three(x3)
            # print('size after down_three: {}'.format(x3_dw.size()))
            
            x4_dw = self.ae_down_four(x3_dw)
            # print('size after down_four: {}'.format(x4_dw.size()))

            
            x4_code = self.ae_fc1(x4_dw.view(x4_dw.size(0), self.nfilters*16*16))
            # this is the encoding for the input. We can retrieve this output 
            # through forward hook, see below.
            
        if (interpret_x == -1): # encode only
            return x4_code
        
        if (interpret_x == 1): 
            # decode only, so define x3_code to be the input x
            x4_code = x
            
        # Given the previous two ifs, if encode only we never get here,
        # and otherwise we want to go ahead with this part, whether x3_code 
        # was defined as the output of fc1 or the input argument x itself.
        
        
        x4_full = self.ae_fc2(x4_code)
        # print('size after fc2: {}'.format(x4_full.size()))
        x4_full = x4_full.view(x4_full.size(0), 1, 16, 16) # 4*self.nfilters
        # print('size after view of fc2: {}'.format(x4_full.size()))
        
        x4_up = self.ae_up_four(x4_full)
        x4 = self.ae_block_four(x4_up)
        # print('size after up_four and block_four: {}'.format(x4.size()))
        
        x3_up = self.ae_up_three(x4)
        x3 = self.ae_block_five(x3_up)
        # print('size after up_three and block_five: {}'.format(x3.size()))
        
        x2_up = self.ae_up_two(x3)
        x2 = self.ae_block_six(x2_up)
        # print('size after up_two and block_six: {}'.format(x2.size()))
        
        x1_up = self.ae_up_one(x2)
        # print('size after up_one: {}'.format(x1_up.size()))
        
        ae_out = self.ae_out_conv(x1_up)
        # print('out size after last conv: {}'.format(out.size()))
        
        out = self.ae_final_sig(ae_out)
        
        return out
    
    
'''
Anatomically-constrained CNN. This is designed after the [Oktay et al.](https://arxiv.org/abs/1705.08302)
paper, which the CAMUS, [LeClerc et al., TMI 2019](https://www.creatis.insa-lyon.fr/~bernard/publis/tmi_2019_leclerc.pdf)
people tried out also.
'''

class ACNNet(nn.Module):
    def __init__(self, 
                 ae_filename, ae_encoding_dim, ae_initial_filters, 
                 n_channels, n_classes,
                 n_filters=16, normalization='none'):
        super(ACNNet, self).__init__()
        
        self.net_ae = AutoEncoder(encoding_dim=ae_encoding_dim, nfilters=ae_initial_filters)
        self.net_ae = torch.nn.DataParallel(self.net_ae)
        
        self.net_ae.load_state_dict(\
            torch.load(os.path.join('src', 'saved_models', ae_filename)))
        
        # Freeze the learning on the ae.
        for param in self.net_ae.parameters():
            param.requires_grad = False
        # This is apparently faster.
        self.net_ae.eval()
        

        if n_channels > 1:
            self.block_one = ResidualConvBlock(1, n_channels, n_filters, normalization=normalization, expand_chan=True)
        else:
            self.block_one = ResidualConvBlock(1, n_channels, n_filters, normalization=normalization)

        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ResidualConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ResidualConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ResidualConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ResidualConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ResidualConvBlock(1, n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv2d(n_filters, n_classes, 1, padding=0)
        
        self.final_sig = torch.nn.Sigmoid()
        
        # Cross entroy loss obviates the need to use this
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
#         self.softmax = nn.Softmax2d(dim=0)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4 # Here's where we could try concat instead.

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)

        f_out = self.out_conv(x9)
        
        unet_out = self.final_sig(f_out)
        
        
        # Now get the AutoEncoder's output (just the encoding part) on the result as well.
        ae_encoded = self.net_ae(unet_out, -1)
        

        return unet_out, ae_encoded
    
    

'''
Here I'm constructing the original UNet to compare against. 
I've ripped various PyTorch examples for this:
https://github.com/usuyama/pytorch-unet

There's a bit of an issue compared to the 'original' UNet in
that in the original there was cropping due to using only the
valid pixels in any convolution. The default behavior now
produces feature maps that are the same size as the input,
through padding=1.

I keep that for simplicity's sake.

'''

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        # If data parallel, then the batch size is split across the number of GPUS. 
        # I was confused because a bs of 16 was getting output of 8 here. 
        # print('forward: input is size {}'.format(x.size()))
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


'''    
This is a U-net I've modified to look more like the
LeClerc unet, which has batch norm, dropout, etc. At least 
what I think is that architecture
I'm borrowing from
https://github.com/zhixuhao/unet/blob/master/model.py
and
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8649738

'''

def double_conv_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   


class UNet_LeClerc(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv_bn(1, 32)
        self.dconv_down2 = double_conv_bn(32, 64)
        self.dconv_down3 = double_conv_bn(64, 128)
        self.dconv_down4 = double_conv_bn(128, 256)  
        self.dconv_down5 = double_conv_bn(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    
        # LeClerc claim upsampling by deconvolution
        
        self.upsample4 = UpConv(512, 256)
        self.upsample3 = UpConv(256, 128)
        self.upsample2 = UpConv(128, 64)
        self.upsample1 = UpConv(64, 32)
        
        self.dconv_up4 = double_conv_bn(256 + 256, 256)
        self.dconv_up3 = double_conv_bn(128 + 128, 128)
        self.dconv_up2 = double_conv_bn(64 + 64, 64)
        self.dconv_up1 = double_conv_bn(32 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, n_class, 1)
        
        
    def forward(self, x):
        # If data parallel, then the batch size is split across the number of GPUS. 
        # I was confused because a bs of 16 was getting output of 8 here. 
        # print('forward: input is size {}'.format(x.size()))
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1) # 128x128

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2) # 64x64
        
        conv3 = self.dconv_down3(x) # 64 -> 128
        x = self.maxpool(conv3) # 32x32
        
        conv4 = self.dconv_down4(x)  # 128 -> 256
        x = self.maxpool(conv4) # 16x16
        
        x = self.dconv_down5(x)  # 256 -> 512 features
        
        x = self.upsample4(x) # returns 256 features.  32x32     
        x = torch.cat([x, conv4], dim=1)  # added to the 256 from the encoding path
        x = self.dconv_up4(x) # returns 256
        
        x = self.upsample3(x) # returns 128 features, 64x64
        x = torch.cat([x, conv3], dim=1) # added to the 128...
        x = self.dconv_up3(x)
        
        
        x = self.upsample2(x)   # returns 64 features  128x128 
        x = torch.cat([x, conv2], dim=1)  # added to the 64 ...
        x = self.dconv_up2(x)
        
        
        x = self.upsample1(x)  # returns 32 features 256x256     
        x = torch.cat([x, conv1], dim=1)   
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out