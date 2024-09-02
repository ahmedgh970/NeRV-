import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from einops import rearrange



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
        return self.ffn(x) + x
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


# -------------------- PatchEmbed  -----------------------
class PatchEmbed(nn.Module):
    """
    Patch Embedding that allows for images with different height and width.
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        H, W = img_size
        self.p1, self.p2 = patch_size
        
        # Check if the image dimensions are divisible by the patch dimensions
        if H % self.p1 != 0 or W % self.p2 != 0:
            raise ValueError(f"Image dimensions ({H, W}) must be divisible by the patch dimensions ({self.p1, self.p2}).")
        
        self.h, self.w = H//self.p1, W//self.p2
        self.N = self.h * self.w

        #self.proj = nn.Linear(in_chans * self.p1 * self.p2, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) 
        # Rearrange and flatten the patches
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p1, p2=self.p2)
        # Apply the linear projection to get (B, N, E)
        #x = self.proj(x)
        return x

def generate_tuples(strides, patch_size):
    p1, p2 = patch_size
    patch_sizes = [(p1, p2)]
    cum_prod = 1
    for num in strides[:-1]:
        cum_prod *= num
        patch_sizes.append((cum_prod * p1, cum_prod * p2))
    return patch_sizes
# --------------------------------------------------------