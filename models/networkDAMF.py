# 2022/2/27 15:30
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from models.unet_parts import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, ws=2):
        super(Attention, self).__init__()
        self.heads = num_heads
        self.ws = ws
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 512*
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 512*3
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 512
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.scale =  num_heads ** -0.5



    def forward(self, x, shuffle):
        if shuffle:
            x = self.shuffle_data(x, 4)

        qkv = self.qkv_dwconv(self.qkv(x))

        attn = self.MSA(qkv)
        # q, k, v = qkv.chunk(3, dim=1)

        # attn = self.attention_opt(q, k, v, self.num_heads, 4)
        out = self.project_out(attn)
        if shuffle:
            out = self.shuffle_back(out, 4)
        return out

    def shuffle_data(self, x, block):
        B, C, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if h % block != 0 or w % block != 0:
            raise ValueError(f'Feature map size {(h, w)} not divisible by block ({block})')
        x = x.reshape(-1, block, int(h // block),
                      block, int(w // block), C)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, h * w, C)
        x = rearrange(x, "b (h w) c -> b c h w", h=h)
        return x

    def shuffle_back(self, x, block):
        B, C, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x.reshape(-1, int(h // block), int(w // block),
                      block, block, C)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, h * w, C)
        x = rearrange(x, "b (h w) c -> b c h w", h=h)
        return x

    def MSA(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H, W, C)
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = x.reshape(B, total_groups, -1, 3, self.heads, (C // 3) // self.heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C // 3)
        x = attn.transpose(2, 3).reshape(B, C // 3, h_group * self.ws, w_group * self.ws)
        return x

class Block(nn.Module):
    def __init__(self,
                 input_dim,
                 dim,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 num_heads=4,
                 LayerNorm_type='WithBias'
                 ):
        super(Block, self).__init__()
        self.inp_dim = input_dim
        self.norm1 = LayerNorm(dim=int(input_dim / 2), LayerNorm_type=LayerNorm_type)
        self.ffn = FeedForward(dim=int(input_dim / 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.norm2 = LayerNorm(dim=int(input_dim / 2), LayerNorm_type=LayerNorm_type)
        self.att = Attention(dim=int(input_dim / 2), num_heads=num_heads, bias=bias)

        self.up = up(input_dim, dim)

    def forward(self, input, enc_out, shuffle):

        att_op = input + self.att(self.norm2(input), shuffle)
        att_op = att_op + self.ffn(self.norm1(att_op))

        att_op = self.up(att_op, enc_out)
        return att_op

class DAM_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DAM_Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(True)
        )

    def forward(self, x):
        for i in range(len(self.classifier)):
            x = self.classifier[i](x)
            if i == 3:
                x_c = x
        b, _, h, _ = x.shape
        x = nn.MaxPool2d(kernel_size=h, stride=h)(x)
        return x.view(b, -1), x_c


class DAM_fusion(nn.Module):
    def __init__(self):
        super(DAM_fusion, self).__init__()
        self.up = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(1024, 512, 1, 1)

    def forward(self, x, x_c):
        x_c = self.up(x_c)
        x = torch.cat([x, x_c], dim=1)
        x = self.conv(x)

        return x


class DAMFDecoder_l(nn.Module):
    def __init__(self,
                 n_channels=3
                 ):
        super(DAMFDecoder_l, self).__init__()

        self.block1 = Block(1024, 256)
        self.block2 = Block(512, 128)
        self.block3 = Block(256, 64)
        self.block4 = Block(128, 64)
        self.outc = outconv(64, n_channels)
        self.sigmod = nn.Sigmoid()
        self.DAM_fusion = DAM_fusion()


    def forward(self, enc_out, enc_outs, I_c, fusion):
        input = self.sigmod(enc_out)
        if fusion:
            input = self.DAM_fusion(input, I_c)

        x = self.block1(input, enc_outs[3], True)  # [4, 256, 32, 32]
        x = self.block2(x, enc_outs[2], True)
        x = self.block3(x, enc_outs[1], True)
        x = self.block4(x, enc_outs[0], True)
        x = self.outc(x)

        return nn.Tanh()(x), input


class DAMFDecoder_h(nn.Module):
    def __init__(self, n_channels=3):
        super(DAMFDecoder_h, self).__init__()
        self.block1 = Block(1024, 256)
        self.block2 = Block(512, 128)
        self.block3 = Block(256, 64)
        self.block4 = Block(128, 64)
        self.outc = outconv(64, n_channels)

    def forward(self, input, enc_outs):
        x = self.block1(input, enc_outs[3], False)
        x = self.block2(x, enc_outs[2], False)
        x = self.block3(x, enc_outs[1], False)
        x = self.block4(x, enc_outs[0], False)
        x = self.outc(x)

        return nn.Tanh()(x)


class DAMFEncoder(nn.Module):
    def __init__(self, n_channels=3):
        super(DAMFEncoder, self).__init__()
        self.inc = inconv(n_channels, 64)  # (conv => BN => ReLU) * 2
        self.down1 = down(64, 128)  # maxpooling+卷积
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5, (x1, x2, x3, x4)


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.para = nn.Parameter(torch.ones(3, 1, 1), requires_grad=True)


    def forward(self, input_l, input_h):
        x = torch.mul(input_l, (1 - self.para)) + torch.mul(input_h, self.para)

        return nn.Tanh()(x)

# enc_out, enc_outs, I_c, fusion
# if __name__ == '__main__':
#     x = torch.rand([1, 3, 256, 256])
#     fusion = False
#     encoder = DAMFEncoder(3)
#     decoder = DAMFDecoder_l(3)
#     decoder_h = DAMFDecoder_h(3)
#     classfilier = DAM_Classifier(num_classes=6)
#     fusion = Fusion()
#     enc_out, enc_outs = encoder(x)
#     I, I_c = classfilier(enc_out)
#     x_l, x_Hinput = decoder(enc_out, enc_outs, I_c, fusion)
#     x_h = decoder_h(x_Hinput, enc_outs)
#     x_enh = fusion(x_l, x_h)
#
#     print(x_enh.shape)