import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import nn, Tensor
import math
import numpy as np
from functools import partial
from torch.nn import init
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch.autograd import Variable
from torchinfo import summary
from typing import Optional, Tuple
from torchvision.extension import _assert_has_ops
from torch.nn.parameter import Parameter
from itertools import repeat
from einops import rearrange
import numbers
import collections.abc

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
def get_seqlen_and_mask(input_resolution, window_size, device):
    attn_map = F.unfold(
        torch.ones([1, 1, input_resolution[0], input_resolution[1]], device=device),
        kernel_size=window_size,
        dilation=1,
        padding=(window_size // 2, window_size // 2),
        stride=1
    )
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask

def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0).unsqueeze(0), key_size[0]).squeeze(0).squeeze(0)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0).unsqueeze(0), key_size[1]).squeeze(0).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)
    axis_kh = axis_kh.reshape(-1)
    axis_kw = axis_kw.reshape(-1)
    axis_qh = axis_qh.reshape(-1)
    axis_qw = axis_qw.reshape(-1)
    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)
    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)
    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0
    ) / torch.log2(torch.tensor(8, dtype=torch.float32))
    return idx_map, relative_coords_table

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=3, stride=1,
            padding=1, bias=True,
            groups=dim
        )
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

from torchvision.extension import _assert_has_ops
class DeformConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,kernel_size)
        self.stride = (stride,stride)
        self.padding = (padding,padding)
        self.dilation = (dilation,dilation)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return deform_conv2d(
            input, offset, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, mask=mask
        )

_pair = _ntuple(2)
def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    _assert_has_ops()
    out_channels = weight.shape[0]
    use_mask = mask is not None
    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = input.shape
    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            "Got offset.shape[1]={}, while 2 * weight.size[2] * weight.size[3]={}".format(
                offset.shape[1], 2 * weights_h * weights_w))

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,)

class DWConv2d_BN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        act_layer=nn.Hardswish,
        offset_clamp=(-1,1)
    ):
        super().__init__()
        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False, groups=in_ch),
            nn.Conv2d(in_ch, 18, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.pwconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0/n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)
        if self.offset_clamp:
            offset = torch.clamp(
                offset, min=self.offset_clamp[0], max=self.offset_clamp[1]
            )
        x = self.dcn(x, offset)
        x = self.pwconv(x)
        x = self.act(x)
        return x

class DWCPatchEmbed(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=24,
                 patch_size=3,
                 stride=1,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1,1)):
        super().__init__()
        self.patch_conv = DWConv2d_BN(
            in_ch = in_chans,
            out_ch = embed_dim,
            kernel_size = patch_size,
            stride = stride,
            act_layer = act_layer,
            offset_clamp = offset_clamp
        )
    def forward(self, x):
        return self.patch_conv(x)

class Patch_Embed_stage(nn.Module):
    def __init__(self, in_chans, embed_dim, num_path=4, offset_clamp=(-1,1)):
        super().__init__()
        self.num_path = num_path
        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans = in_chans,
                embed_dim = embed_dim,
                patch_size = 3,
                stride = 1,
                offset_clamp = offset_clamp
            )
            for _ in range(num_path)
        ])
    def forward(self, x):
        out_list = []
        for pe in self.patch_embeds:
            out_list.append(pe(x))
        return out_list

class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.trained_H, self.trained_W = input_resolution
        self.trained_len = self.trained_H * self.trained_W
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_head
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads,1,1)/0.24).exp()-1)
        )
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_heads,1,self.head_dim),
                mean=0, std=0.02
            )
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cpb_fc1 = nn.Linear(2,512,bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512,num_heads,bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, -1, 3*self.num_heads, self.head_dim
        ).permute(0,2,1,3)
        q, k, v = qkv.chunk(3, dim=1)
        rel_bias = self.cpb_fc2(
            self.cpb_act(self.cpb_fc1(relative_coords_table))
        ).transpose(0,1)[:, relative_pos_index.view(-1)]
        rel_bias = rel_bias.view(-1, self.trained_len, self.trained_len)
        rel_bias = rel_bias.reshape(-1, self.trained_len, self.trained_H, self.trained_W)
        rel_bias = F.interpolate(rel_bias, (H,W), mode='bilinear')
        rel_bias = rel_bias.reshape(-1, self.trained_len, N).transpose(-1,-2).reshape(-1, N, self.trained_H, self.trained_W)
        rel_bias = F.interpolate(rel_bias, (H,W), mode='bilinear').reshape(-1, N, N).transpose(-1,-2)
        attn = (
            (F.normalize(q,dim=-1)+ self.query_embedding) * F.softplus(self.temperature)* seq_length_scale
        ) @ F.normalize(k,dim=-1).transpose(-2,-1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.trained_H, self.trained_W = input_resolution
        self.trained_len = self.trained_H * self.trained_W
        self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[
            1] // self.sr_ratio
        self.trained_pool_len = self.trained_pool_H * self.trained_pool_W
        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        B, N, C = x.shape
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        attn_local = (q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) + self.relative_pos_bias_local.unsqueeze(1)
        if padding_mask is not None:
            attn_local = attn_local.masked_fill(padding_mask, float('-inf'))
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)
        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)
        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                    relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_pool_len)
        pool_bias = pool_bias.reshape(-1, self.trained_len, self.trained_pool_H, self.trained_pool_W)
        pool_bias = F.interpolate(pool_bias, (pool_H, pool_W), mode='bilinear')
        pool_bias = pool_bias.reshape(-1, self.trained_len, pool_len).transpose(-1, -2).reshape(-1, pool_len,
                                                                                                self.trained_H,
                                                                                                self.trained_W)
        pool_bias = F.interpolate(pool_bias, (H, W), mode='bilinear').reshape(-1, pool_len, N).transpose(-1, -2)
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SKFF1(nn.Module):
    def __init__(self,
                 in_channels: int,
                 height: int = 2,
                 mode: str = "bcf",
                 reduction: int = 8,
                 bias: bool = False):
        super().__init__()
        assert mode in {"bcf", "concat"}, "mode must be 'bcf' or 'concat'"
        self.mode   = mode
        self.height = height
        self.C      = in_channels
        d = max(int(in_channels / reduction), 4)

        if mode == "bcf":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_du  = nn.Sequential(
                nn.Conv2d(in_channels, d, 1, bias=bias),
                nn.PReLU()
            )
            self.fcs = nn.ModuleList([
                nn.Conv2d(d, in_channels, 1, bias=bias)
                for _ in range(self.height)
            ])
            self.softmax = nn.Softmax(dim=1)

        else:
            self.conv_reduce = nn.Sequential(
                nn.Conv2d(in_channels * height, in_channels, 1, bias=bias),
                nn.ReLU(inplace=True)
            )

    def forward(self, inp_feats: List[torch.Tensor]) -> torch.Tensor:
        aligned = []
        tgt_sz  = inp_feats[0].shape[2:]
        for f in inp_feats:
            if f.shape[2:] != tgt_sz:
                f = F.interpolate(f, size=tgt_sz, mode="bilinear",
                                  align_corners=False)
            aligned.append(f)
        if self.mode == "concat":
            x = torch.cat(aligned, dim=1)
            return self.conv_reduce(x)
        feats   = torch.stack(aligned, dim=1)
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        att_vec = [fc(feats_Z) for fc in self.fcs]
        att_vec = torch.stack(att_vec, dim=1)
        att_vec = self.softmax(att_vec)
        feats_V = torch.sum(feats * att_vec, dim=1)
        return feats_V

class SKFFcat(nn.Module):
    def __init__(self, in_channels_list, out_channels, reduction=8, bias=False):
        super().__init__()
        self.height = len(in_channels_list)
        d = max(int(out_channels / reduction), 4)
        self.out_channels = out_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.input_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=bias)
            for in_ch in in_channels_list
        ])
        self.conv_du = nn.Sequential(
            nn.Conv2d(out_channels, d, kernel_size=1, bias=bias),
            nn.PReLU()
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(d,out_channels,kernel_size=1,bias=bias)
            for _ in range(self.height)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats:List[torch.Tensor])->torch.Tensor:
        aligned_feats=[]
        target_size = inp_feats[0].shape[2:]
        for idx,feat in enumerate(inp_feats):
            feat = self.input_convs[idx](feat)
            if feat.shape[2:]!=target_size:
                feat=F.interpolate(feat,size=target_size,mode='bilinear',align_corners=False)
            aligned_feats.append(feat)
        inp_feats = torch.stack(aligned_feats,dim=1)
        feats_U = torch.sum(inp_feats,dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors=[fc(feats_Z) for fc in self.fcs]
        attention_vectors=torch.stack(attention_vectors,dim=1)
        attention_vectors=self.softmax(attention_vectors)
        feats_V=torch.sum(inp_feats*attention_vectors,dim=1)
        return feats_V

class DWConv2d_BN_Down(nn.Module):
    def __init__(self, in_ch,out_ch,kernel_size=3,stride=2, offset_clamp=(-1,1),
                 act_layer=nn.Hardswish):
        super().__init__()
        self.offset_clamp=offset_clamp
        self.offset_generator=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=stride,padding=1,bias=False,groups=in_ch),
            nn.Conv2d(in_ch,18,kernel_size=1,stride=1,bias=False)
        )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.pwconv=nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1,bias=False)
        self.act=act_layer()

    def forward(self,x):
        offset=self.offset_generator(x)
        if self.offset_clamp:
            offset=torch.clamp(offset,self.offset_clamp[0],self.offset_clamp[1])
        x=self.dcn(x,offset)
        x=self.pwconv(x)
        x=self.act(x)
        return x

class DWConv2d_BN_Up(nn.Module):
    def __init__(self,in_ch,out_ch,factor=2,offset_clamp=(-1,1),act_layer=nn.Hardswish):
        super().__init__()
        self.factor=factor
        self.offset_clamp=offset_clamp
        self.act=act_layer()
        self.offset_generator=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=1,bias=False,groups=in_ch),
            nn.Conv2d(in_ch,18,kernel_size=1,stride=1,bias=False),
        )
        self.deform_conv0=DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch*4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.deform_conv=DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.pwconv=nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1,bias=False)

    def forward(self,x):
        offset=self.offset_generator(x)
        if self.offset_clamp:
            offset=torch.clamp(offset,self.offset_clamp[0],self.offset_clamp[1])
        x=self.deform_conv0(x,offset)
        x=F.pixel_shuffle(x,self.factor)
        offset=self.offset_generator(x)
        if self.offset_clamp:
            offset=torch.clamp(offset,self.offset_clamp[0],self.offset_clamp[1])
        x=self.deform_conv(x,offset)
        x=self.pwconv(x)
        x=self.act(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn=Attention(
            dim, input_resolution, num_heads,
            qkv_bias, attn_drop, drop
        )
        if sr_ratio==1:
            self.attn=Attention(
                dim, input_resolution, num_heads,
                qkv_bias, attn_drop, drop
            )
        else:
            self.attn=AggregatedAttention(
                dim, input_resolution,
                num_heads=num_heads, window_size=window_size,
                qkv_bias=qkv_bias, attn_drop=attn_drop,
                proj_drop=drop, sr_ratio=sr_ratio
            )
        self.norm2=norm_layer(dim)
        hidden_dim=int(dim*mlp_ratio)
        self.mlp=ConvolutionalGLU(dim,hidden_dim,act_layer=act_layer,drop=drop)
        self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()

    def forward(self,x,H,W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        x_ = self.norm1(x)
        attn_out = self.attn(x_, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask)
        x = x + self.drop_path(attn_out)
        x_ = self.norm2(x)
        mlp_out = self.mlp(x_,H,W)
        x = x + self.drop_path(mlp_out)
        return x

def get_list_dimension(my_list):
    if isinstance(my_list, list):
        return 1 + max(get_list_dimension(item) for item in my_list)
    else:
        return 0

class ResidualBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out

class ResizingResBlock(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=8, num_blocks=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.head_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels=hidden_channels, kernel_size=3))
        self.resblocks = nn.Sequential(*blocks)
        self.tail_conv = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
    
    def forward(self, x, H_out, W_out):
        x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
        feat = self.head_conv(x)
        feat = self.resblocks(feat)
        out  = self.tail_conv(feat)
        return out

class BVIFormer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 img_size=256,
                 pretrain_size=None,
                 depths=[2,2,2],
                 updepths=[2,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[3, 3, 3, 3],
                 mlp_ratios=[8,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True,
                 branch_num = 2,
                 fusion_mode = "bcf",
                 ):
        super().__init__()
        pretrain_size = pretrain_size or img_size
        self.img_size = img_size
        self.depths= depths
        self.updepths= updepths
        self.window_size= window_size
        self.sr_ratios= sr_ratios
        self.so_ratios= so_ratios
        self.embed_dim= embed_dims
        self.branch_num  = branch_num
        self.fusion_mode = fusion_mode.lower()
        total_depth = sum(depths)+sum(updepths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur=0
        rel_idx1, rel_coords1 = get_relative_position_cpb(
            (img_size//4, img_size//4),
            (img_size//4//so_ratios[0], img_size//4//so_ratios[0]),
            (pretrain_size//4, pretrain_size//4)
        )
        self.register_buffer("rel_idx1", rel_idx1, persistent=False)
        self.register_buffer("rel_coords1", rel_coords1, persistent=False)
        self.patch_embed1 = Patch_Embed_stage(in_chans=3, embed_dim=48, num_path=2, offset_clamp=offset_clamp)
        block_list1=[]
        for i_block in range(depths[0]):
            blk=Block(
                dim=embed_dims[0],
                num_heads=2,
                input_resolution=(img_size//4, img_size//4),
                window_size=window_size[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur+i_block],
                sr_ratio=sr_ratios[0]
            )
            block_list1.append(blk)
        self.block1 = nn.ModuleList(block_list1)
        cur+=depths[0]
        self.norm1 = nn.LayerNorm(48)
        self.aggregate1 = SKFF1(in_channels=96, height=2, mode="concat")
        self.down1 = DWConv2d_BN_Down(in_ch=48, out_ch=96, stride=2, offset_clamp=offset_clamp)
        rel_idx2, rel_coords2 = get_relative_position_cpb(
            (img_size//8, img_size//8),
            (img_size//8//so_ratios[1], img_size//8//so_ratios[1]),
            (pretrain_size//8, pretrain_size//8)
        )
        self.register_buffer("rel_idx2", rel_idx2, persistent=False)
        self.register_buffer("rel_coords2", rel_coords2, persistent=False)
        self.patch_embed2 = Patch_Embed_stage(in_chans=96, embed_dim=96, num_path=2, offset_clamp=offset_clamp)
        block_list2=[]
        for i_block in range(depths[1]):
            blk=Block(
                dim=embed_dims[1],
                num_heads=4,
                input_resolution=(img_size//8, img_size//8),
                window_size=window_size[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur+i_block],
                sr_ratio=sr_ratios[1]
            )
            block_list2.append(blk)
        self.block2 = nn.ModuleList(block_list2)
        cur+=depths[1]
        self.norm2 = nn.LayerNorm(96)
        self.aggregate2 = SKFF1(in_channels=384, height=2, mode="concat")
        self.down2 = DWConv2d_BN_Down(in_ch=96, out_ch=384, stride=2, offset_clamp=offset_clamp)
        rel_idx3, rel_coords3 = get_relative_position_cpb(
            (img_size//16, img_size//16),
            (img_size//16//so_ratios[2], img_size//16//so_ratios[2]),
            (pretrain_size//16, pretrain_size//16)
        )
        self.register_buffer("rel_idx3", rel_idx3, persistent=False)
        self.register_buffer("rel_coords3", rel_coords3, persistent=False)
        self.patch_embed3 = Patch_Embed_stage(in_chans=384, embed_dim=384, num_path=2, offset_clamp=offset_clamp)
        block_list3=[]
        for i_block in range(depths[2]):
            blk=Block(
                dim=384,
                num_heads=8,
                input_resolution=(img_size//16, img_size//16),
                window_size=window_size[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur+i_block],
                sr_ratio=sr_ratios[2]
            )
            block_list3.append(blk)
        self.block3 = nn.ModuleList(block_list3)
        cur+=depths[2]
        self.norm3 = nn.LayerNorm(384)
        self.aggregate3 = SKFF1(in_channels=1536, height=2, mode="concat")
        self.down3 = DWConv2d_BN_Down(in_ch=384, out_ch=1536, offset_clamp=offset_clamp)
        up_rel_idx1, up_rel_coords1 = get_relative_position_cpb(
            (img_size//16, img_size//16),
            (img_size//16//so_ratios[0], img_size//16//so_ratios[0]),
            (pretrain_size//16, pretrain_size//16)
        )
        self.register_buffer("up_rel_idx1", up_rel_idx1, persistent=False)
        self.register_buffer("up_rel_coords1", up_rel_coords1, persistent=False)
        self.up_patch_embed1 = Patch_Embed_stage(in_chans=1536, embed_dim=384, num_path=2, offset_clamp=offset_clamp)
        upblock_list1=[]
        for i_block in range(updepths[0]):
            blk=Block(
                dim=384,
                num_heads=8,
                input_resolution=(img_size//16,img_size//16),
                window_size=window_size[0],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur+i_block],
                sr_ratio=sr_ratios[0]
            )
            upblock_list1.append(blk)
        self.up_block1=nn.ModuleList(upblock_list1)
        cur+=updepths[0]
        self.up_norm1 = nn.LayerNorm(384)
        self.up_aggregate1 = SKFF1(in_channels=384, height=2, mode="concat")
        self.up_module1 = DWConv2d_BN_Up(in_ch=384, out_ch=384, factor=2, offset_clamp=offset_clamp)
        self.skff_cat1=SKFFcat(
            in_channels_list=[384, 1536],
            out_channels=384
        )
        up_rel_idx2, up_rel_coords2 = get_relative_position_cpb(
            (img_size//8, img_size//8),
            (img_size//8//so_ratios[1], img_size//8//so_ratios[1]),
            (pretrain_size//8, pretrain_size//8)
        )
        self.register_buffer("up_rel_idx2", up_rel_idx2, persistent=False)
        self.register_buffer("up_rel_coords2", up_rel_coords2, persistent=False)
        self.up_patch_embed2 = Patch_Embed_stage(in_chans=384, embed_dim=96, num_path=2, offset_clamp=offset_clamp)
        upblock_list2=[]
        for i_block in range(updepths[1]):
            blk=Block(
                dim=96,
                num_heads=4,
                input_resolution=(img_size//8,img_size//8),
                window_size=window_size[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur+i_block],
                sr_ratio=sr_ratios[1]
            )
            upblock_list2.append(blk)
        self.up_block2=nn.ModuleList(upblock_list2)
        cur+=updepths[1]
        self.up_norm2 = nn.LayerNorm(96)
        self.up_aggregate2 = SKFF1(in_channels=96, height=2, mode="concat")
        self.up_module2 = DWConv2d_BN_Up(in_ch=96, out_ch=96, factor=2, offset_clamp=offset_clamp)
        self.skff_cat2=SKFFcat(
            in_channels_list=[96, 384],
            out_channels=96
        )
        up_rel_idx3, up_rel_coords3 = get_relative_position_cpb(
            (img_size//4, img_size//4),
            (img_size//4//so_ratios[2], img_size//4//so_ratios[2]),
            (pretrain_size//4, pretrain_size//4)
        )
        self.register_buffer("up_rel_idx3", up_rel_idx3, persistent=False)
        self.register_buffer("up_rel_coords3", up_rel_coords3, persistent=False)
        self.up_patch_embed3 = Patch_Embed_stage(in_chans=96, embed_dim=48, num_path=2, offset_clamp=offset_clamp)
        upblock_list3=[]
        for i_block in range(updepths[2]):
            blk=Block(
                dim=48,
                num_heads=2,
                input_resolution=(img_size//4,img_size//4),
                window_size=window_size[2],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur+i_block],
                sr_ratio=sr_ratios[2]
            )
            upblock_list3.append(blk)
        self.up_block3=nn.ModuleList(upblock_list3)
        cur+=updepths[2]
        self.up_norm3 = nn.LayerNorm(48)
        self.up_aggregate3 = SKFF1(in_channels=48, height=2, mode="concat")
        self.up_module3 = DWConv2d_BN_Up(in_ch=48, out_ch=48, factor=2, offset_clamp=offset_clamp)
        self.output_conv = nn.Conv2d(48, 3, kernel_size=3, stride=1, padding=1, bias=False)
        for n,m in self.named_modules():
            self._init_weights(m,n)
        self.final_resize = final_resize
        if final_resize:
            self.resize_module = ResizingResBlock(in_channels=3, hidden_channels=4, num_blocks=2)

    def _init_weights(self,m:nn.Module,name:str=''):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out//=m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,(nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        _, _, H_input, W_input = x.size()
        x1 = self.forward_stage1(x)
        x2 = self.forward_stage2(x1)
        x3 = self.forward_stage3(x2)
        x_up1 = self.forward_upstage1(x3)
        x_up1= self.skff_cat1([x_up1, x3])
        x_up2 = self.forward_upstage2(x_up1)
        x_up2= self.skff_cat2([x_up2, x2])
        x_up3 = self.forward_upstage3(x_up2)
        out = self.output_conv(x_up3)
        return out

    def forward_stage1(self, x):
        x_list1 = self.patch_embed1(x)
        out_list1 = []
        for feat in x_list1:
            B_, C_, H_, W_ = feat.shape
            y = feat.flatten(2).transpose(1, 2)
            local_seq_length, padding_mask = get_seqlen_and_mask((H_, W_), 3, y.device)
            seq_length_scale = 1.0
            for blk in self.block1:
                y = blk(y, H_, W_, self.rel_idx1, self.rel_coords1, seq_length_scale, padding_mask)
            y = self.norm1(y)
            y = y.transpose(1, 2).reshape(B_, C_, H_, W_)
            y = self.down1(y)
            out_list1.append(y)
        x1 = self.aggregate1(out_list1)
        skip1 = x1
        return x1

    def forward_stage2(self, x1):
        x_list2 = self.patch_embed2(x1)
        out_list2 = []
        for feat in x_list2:
            B_, C_, H_, W_ = feat.shape
            y = feat.flatten(2).transpose(1, 2)
            local_seq_length, padding_mask = get_seqlen_and_mask((H_, W_), 3, y.device)
            seq_length_scale = 1.0
            for blk in self.block2:
                y = blk(y, H_, W_, self.rel_idx2, self.rel_coords2, seq_length_scale, padding_mask)
            y = self.norm2(y)
            y = y.transpose(1, 2).reshape(B_, C_, H_, W_)
            y = self.down2(y)
            out_list2.append(y)
        x2 = self.aggregate2(out_list2)
        skip2 = x2
        return x2

    def forward_stage3(self, x2):
        x_list3 = self.patch_embed3(x2)
        out_list3 = []
        for feat in x_list3:
            B_, C_, H_, W_ = feat.shape
            y = feat.flatten(2).transpose(1, 2)
            local_seq_length, padding_mask = get_seqlen_and_mask((H_, W_), 3, y.device)
            seq_length_scale = 1.0
            for blk in self.block3:
                y = blk(y, H_, W_, self.rel_idx3, self.rel_coords3, seq_length_scale, padding_mask)
            y = self.norm3(y)
            y = y.transpose(1, 2).reshape(B_, C_, H_, W_)
            y = self.down3(y)
            out_list3.append(y)
        x3 = self.aggregate3(out_list3)
        skip3 = x3
        return x3

    def forward_upstage1(self, x3):
        x_list1 = self.up_patch_embed1(x3)
        dec_out_list1 = []
        for feat in x_list1:
            B_, C_, H_, W_ = feat.shape
            y = feat.flatten(2).transpose(1, 2)
            local_seq_length, padding_mask = get_seqlen_and_mask((H_, W_), 3, y.device)
            seq_length_scale = 1.0
            for blk in self.up_block1:
                y = blk(y, H_, W_, self.up_rel_idx1, self.up_rel_coords1, seq_length_scale, padding_mask)
            y = self.up_norm1(y)
            y = y.transpose(1, 2).reshape(B_, C_, H_, W_)
            y = self.up_module1(y)
            dec_out_list1.append(y)
        x_up1 = self.up_aggregate1(dec_out_list1)
        return x_up1

    def forward_upstage2(self, x_up1):
        x_list2 = self.up_patch_embed2(x_up1)
        dec_out_list2 = []
        for feat in x_list2:
            B_, C_, H_, W_ = feat.shape
            y = feat.flatten(2).transpose(1, 2)
            local_seq_length, padding_mask = get_seqlen_and_mask((H_, W_), 3, y.device)
            seq_length_scale = 1.0
            for blk in self.up_block2:
                y = blk(y, H_, W_, self.up_rel_idx2, self.up_rel_coords2, seq_length_scale, padding_mask)
            y = self.up_norm2(y)
            y = y.transpose(1, 2).reshape(B_, C_, H_, W_)
            y = self.up_module2(y)
            dec_out_list2.append(y)
        x_up2 = self.up_aggregate2(dec_out_list2)
        return x_up2

    def forward_upstage3(self, x_up2):
        x_list3 = self.up_patch_embed3(x_up2)
        dec_out_list3 = []
        for feat in x_list3:
            B_, C_, H_, W_ = feat.shape
            y = feat.flatten(2).transpose(1, 2)
            local_seq_length, padding_mask = get_seqlen_and_mask((H_, W_), 3, y.device)
            seq_length_scale = 1.0
            for blk in self.up_block3:
                y = blk(y, H_, W_, self.up_rel_idx3, self.up_rel_coords3, seq_length_scale, padding_mask)
            y = self.up_norm3(y)
            y = y.transpose(1, 2).reshape(B_, C_, H_, W_)
            y = self.up_module3(y)
            dec_out_list3.append(y)
        x_up3 = self.up_aggregate3(dec_out_list3)
        return x_up3

def bviformer_t():
    return BVIFormer(depths=[2,2,2],
                 updepths=[2,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[3, 3, 3, 3],
                 mlp_ratios=[8,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True)

def bviformer_s():
    return BVIFormer(depths=[2,2,2],
                 updepths=[2,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[3, 3, 3, 3],
                 mlp_ratios=[8,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True,
                 branch_num = 2,
                 fusion_mode = "bcf")

def bviformer_g():
    return BVIFormer(depths=[2,2,2],
                 updepths=[2,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[3, 3, 3, 3],
                 mlp_ratios=[8,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True,
                 branch_num = 2,
                 fusion_mode = "concat")

def bviformer_b():
    return BVIFormer(depths=[2,2,2],
                 updepths=[2,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[3, 3, 3, 3],
                 mlp_ratios=[8,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True)

def bviformer_d():
    return BVIFormer(depths=[2,2,2],
                 updepths=[2,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[3, 3, 3, 3],
                 mlp_ratios=[8,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True,
                 branch_num = 2,
                 fusion_mode = "concat")

def bviformer_w():
    return BVIFormer(depths=[2,2,2],
                 updepths=[2,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[5, 5, 5, 5],
                 mlp_ratios=[8,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True,
                 branch_num = 2,
                 fusion_mode = "bcf")

def bviformer_m():
    return BVIFormer(depths=[2,2,2],
                 updepths=[2,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[5, 5, 3, 3],
                 mlp_ratios=[16,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True,
                 branch_num = 2,
                 fusion_mode = "bcf")

def bviformer_l():
    return BVIFormer(depths=[4,2,2],
                 updepths=[4,2,2],
                 sr_ratios=[16,8,4],
                 so_ratios=[16,8,4],
                 window_size=[5, 5, 5, 5],
                 mlp_ratios=[16,8,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 offset_clamp=(-1,1),
                 embed_dims=[48,96,384],
                 updims=[384,96,48],
                 final_resize=True,
                 branch_num = 2,
                 fusion_mode = "bcf")
dtype= torch.FloatTensor

