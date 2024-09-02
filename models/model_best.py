import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchinfo

import numpy as np
from PIL import Image

from timm.models.layers import trunc_normal_

from models.layers import Gmlp, ResidualFFN, GatedCNNBlock


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