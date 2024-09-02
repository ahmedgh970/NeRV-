import os
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.layers import Attention2d, MultiQueryAttentionV2, MultiQueryAttention2d

from functools import partial
from utils import *
from einops import rearrange, repeat



class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, vid_list=[None], frame_gap=1,  visualize=False):
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()

        num_frame = 0 
        for img_id in all_imgs:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)
            num_frame += 1          

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        accum_img_num.append(num_frame)
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        self.accum_img_num = np.asfarray(accum_img_num)
        if None not in vid_list:
            self.frame_idx = [self.frame_idx[i] for i in vid_list]
        self.frame_gap = frame_gap

    def __len__(self):
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        valid_idx = idx * self.frame_gap
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)
        frame_idx = torch.tensor(self.frame_idx[valid_idx])

        return tensor_image, frame_idx

class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, inpt):
        return torch.sin(inpt)


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


# --------------------- Custom Conv ------------------------
class CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            #trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.up_scale(self.conv(x))
# --------------------------------------------------------
    

# --------------------- G-MLP ------------------------
class Gmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(-1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x
# --------------------------------------------------------


# --------------------- Residual FFN ------------------------
class ResidualFFN(nn.Module):
    def __init__(self, in_features, dropout=0.,):
        super(ResidualFFN, self).__init__()
        self.attn = Attention2d(
            dim=in_features,
            dim_out = None,
            num_heads = 4,
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, groups=in_features),
            nn.Dropout(dropout),
            nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, padding=0),
            nn.Dropout(dropout),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x
# --------------------------------------------------------


# --------------------- GatedCNNBlock ------------------------
class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        shortcut = x # [B, H, W, C]
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        x = x + shortcut
        x = x.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        return x
# --------------------------------------------------------


# --------------------- Conv Block ------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, dropout=0.):
        super(ConvBlock, self).__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = hidden_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.conv_block(x)
# --------------------------------------------------------




class NervPlusBlock(nn.Module):
    def __init__(self, resolution, step, **kargs):
        super(NervPlusBlock, self).__init__()
        self.proj = ConvBlock(in_channels=kargs['ngf'], hidden_channels=kargs['new_ngf'], out_channels=kargs['new_ngf'])

        self.gcnn = GatedCNNBlock(
            kargs['new_ngf'], expansion_ratio=8/3,
            kernel_size=5, conv_ratio=1.0,
            act_layer=nn.GELU,
        )
        
        self.nerv_block = nn.Sequential(
            CustomConv(
                ngf=kargs['new_ngf'], new_ngf=kargs['new_ngf'],
                stride=kargs['stride'], bias=kargs['bias'],
                conv_type=kargs['conv_type']
            ),
            ActivationLayer(kargs['act']),
            nn.Dropout(0.),
        )
        
        self.rffn = ResidualFFN(in_features=kargs['new_ngf'])
        
        self.stride = kargs['stride']

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.proj(x)
        res = x
        x = self.rffn(self.nerv_block(self.gcnn(x)))
        x += F.interpolate(res, scale_factor=self.stride, mode='bilinear')
        return x


class Generator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.stem = Gmlp(
            in_features=kargs['embed_length'],
            hidden_features=stem_dim,
            out_features=self.fc_h * self.fc_w * self.fc_dim,
            act_layer=approx_gelu,
            drop=0.,
        )

        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
                fc_hw = (self.fc_h * stride, self.fc_w * stride)
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])
                fc_hw = (fc_hw[0] * stride, fc_hw[1] * stride)

            for j in range(kargs['num_blocks']):
                self.layers.append(NervPlusBlock(resolution=fc_hw, step=i, ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                    bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], conv_type=kargs['conv_type']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if kargs['sin_res']:
                if i == len(kargs['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
                    # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias'])
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
                # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inpt):
        b, l = inpt.size(0), inpt.size(1)

        output = self.stem(inpt).view(b, self.fc_dim, self.fc_h, self.fc_w)
        out_list = []
        for i, (layer, head_layer) in enumerate(zip(self.layers, self.head_layers)):
            output = layer(output)
            if head_layer is not None:
                img_out = head_layer(output)
                # normalize the final output with sigmoid or tanh function
                # in our case we are using (torch.tanh(img_out) + 1) * 0.5
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)
        return  out_list
    


'''
idx = 1
x = torch.randn(idx).to("cuda")
print(x.shape)

PE = PositionalEncoding('1.25_40').to("cuda")
inpt = PE(x)

print(inpt.shape)

model = Generator(
    embed_length=PE.embed_length,
    stem_dim_num='512_1',
    fc_hw_dim='9_16_8',
    expansion=1,
    num_blocks=1,
    bias=True,
    act='swish',
    reduction=2,
    stride_list=[5, 2, 2, 2, 2],
    sin_res=True,
    sigmoid=True,
    lower_width=80,
    conv_type='conv',
    norm='none'
).to("cuda")

out_list = model(inpt)
print('length de out_list: ', len(out_list))
print('shape of out_list[0]: ', out_list[0].shape)

# Calculate the total number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {trainable_params}")
#'''