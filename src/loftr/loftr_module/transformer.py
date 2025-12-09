import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import Attention, crop_feature, pad_feature
from einops.einops import rearrange
from collections import OrderedDict
from ..utils.position_encoding import RoPEPositionEncodingSine
import numpy as np
from loguru import logger    


class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(
            d_model, d_model,
            kernel_size=agg_size0, padding=0, stride=agg_size0,
            bias=False, groups=d_model
        ) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=self.agg_size1, stride=self.agg_size1
        ) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(
                d_model, max_shape=(256, 256), npe=npe, ropefp16=True
            )

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional)
            source_mask (torch.Tensor): [N, H1, W1] (optional)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggregate feature
        query = self.aggregate(x).permute(0, 2, 3, 1)           # [N, H0/agg, W0/agg, C]
        source_agg = self.max_pool(source).permute(0, 2, 3, 1)  # [N, H1/agg, W1/agg, C]
        query = self.norm1(query)
        source_agg = self.norm1(source_agg)

        if x_mask is not None:
            x_mask, source_mask = map(
                lambda m: self.max_pool(m.float()).bool(),
                [x_mask, source_mask]
            )

        query, key, value = self.q_proj(query), self.k_proj(source_agg), self.v_proj(source_agg)

        # Positional encoding
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.merge(m.reshape(bs, -1, self.nhead * self.dim))  # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w',
                      h=H0 // self.agg_size0,
                      w=W0 // self.agg_size0)  # [N, C, H0/agg, W0/agg]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(
                m, scale_factor=self.agg_size0,
                mode='bilinear', align_corners=False
            )  # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1))  # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2)  # [N, C, H0, W0]

        return x + m


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module with random-exit on cross-attention."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        # full loftr config (already lowered)
        self.full_config = config
        self.fp32 = not (config['mp'] or config['half'])

        # coarse-level config
        coarse_cfg = config['coarse']
        self.d_model = coarse_cfg['d_model']
        self.nhead = coarse_cfg['nhead']
        self.layer_names = coarse_cfg['layer_names']
        self.agg_size0, self.agg_size1 = coarse_cfg['agg_size0'], coarse_cfg['agg_size1']
        self.rope = coarse_cfg['rope']

        # random-exit related configs
        self.random_exit = coarse_cfg.get('random_exit', False)
        self.exit_on_cross_only = coarse_cfg.get('exit_on_cross_only', True)
        self.exit_min_block = coarse_cfg.get('exit_min_block', 0)
        self.exit_max_block = coarse_cfg.get('exit_max_block', -1)
        self.save_all_layers = coarse_cfg.get('save_all_layers', False)

        # pre-compute all cross layer indices (each is a block end)
        self.cross_layer_indices = [i for i, name in enumerate(self.layer_names) if name == 'cross']
        self.num_blocks = len(self.cross_layer_indices)

        # build layers
        self_layer = AG_RoPE_EncoderLayer(
            coarse_cfg['d_model'], coarse_cfg['nhead'],
            coarse_cfg['agg_size0'], coarse_cfg['agg_size1'],
            coarse_cfg['no_flash'], coarse_cfg['rope'],
            coarse_cfg['npe'], self.fp32
        )
        cross_layer = AG_RoPE_EncoderLayer(
            coarse_cfg['d_model'], coarse_cfg['nhead'],
            coarse_cfg['agg_size0'], coarse_cfg['agg_size1'],
            coarse_cfg['no_flash'], False,
            coarse_cfg['npe'], self.fp32
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(self_layer) if name == 'self' else copy.deepcopy(cross_layer)
             for name in self.layer_names]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _sample_exit_layer_idx(self, feat0_device):
        """
        根据配置，在 cross-block 上随机采样一个退出层（只在训练 + random_exit 时使用）。
        返回：exit_layer_idx, exit_block_idx（若不启用 random-exit，则都为 None）
        """
        if not (self.training and self.random_exit and self.num_blocks > 0):
            return None, None

        # block 范围裁剪到合法区间
        min_block = max(0, int(self.exit_min_block))
        max_block = int(self.exit_max_block)
        if max_block < 0 or max_block >= self.num_blocks:
            max_block = self.num_blocks - 1
        if min_block > max_block:
            # 防止配置错误，退化成全范围
            min_block, max_block = 0, self.num_blocks - 1

        # 在 [min_block, max_block] 上均匀采样一个 block
        block_idx = torch.randint(
            low=min_block,
            high=max_block + 1,
            size=(1,),
            device=feat0_device
        ).item()
        exit_layer_idx = self.cross_layer_indices[block_idx]
        return exit_layer_idx, block_idx

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, H, W] (optional)
            mask1 (torch.Tensor): [N, H, W] (optional)
        """
        H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]

        # 决定这次前向的退出层（只在训练 + random_exit 时生效）
        exit_layer_idx, exit_block_idx = self._sample_exit_layer_idx(feat0.device)

        # 可选：保存所有层特征（主要用于离线打标签）
        all_feats0, all_feats1 = [], []

        feature_cropped = False
        if bs == 1 and mask0 is not None and mask1 is not None:
            mask_H0, mask_W0 = mask0.size(-2), mask0.size(-1)
            mask_H1, mask_W1 = mask1.size(-2), mask1.size(-1)
            mask_h0 = mask0[0].sum(-2)[0]
            mask_w0 = mask0[0].sum(-1)[0]
            mask_h1 = mask1[0].sum(-2)[0]
            mask_w1 = mask1[0].sum(-1)[0]
            mask_h0 = mask_h0 // self.agg_size0 * self.agg_size0
            mask_w0 = mask_w0 // self.agg_size0 * self.agg_size0
            mask_h1 = mask_h1 // self.agg_size1 * self.agg_size1
            mask_w1 = mask_w1 // self.agg_size1 * self.agg_size1
            feat0 = feat0[:, :, :mask_h0, :mask_w0]
            feat1 = feat1[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if feature_cropped:
                mask0, mask1 = None, None

            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError(f"Unknown layer name: {name}")

            # 可选：记录所有层输出
            if self.save_all_layers and data is not None:
                all_feats0.append(feat0)
                all_feats1.append(feat1)

            # 训练 + random-exit 时，在采样到的 cross 层后直接退出
            if exit_layer_idx is not None and i == exit_layer_idx:
                break

        # 若需要，把所有层的 coarse 特征挂到 data 上（主要用于离线打标签脚本）
        if self.save_all_layers and data is not None:
            data['coarse_feats0_all'] = all_feats0
            data['coarse_feats1_all'] = all_feats1

        # 同样把这次选中的 exit 信息记下来（如果有的话）
        if data is not None:
            if exit_layer_idx is not None:
                data['coarse_exit_layer_idx'] = exit_layer_idx
                if exit_block_idx is not None:
                    data['coarse_exit_block_idx'] = exit_block_idx
            else:
                # eval 或未开启 random_exit：统一用最后一层
                data['coarse_exit_layer_idx'] = len(self.layer_names) - 1

        if feature_cropped:
            # padding feature 回原尺寸
            bs, c, mask_h0, mask_w0 = feat0.size()
            if mask_h0 != H0:
                feat0 = torch.cat(
                    [feat0,
                     torch.zeros(bs, c, H0 - mask_h0, W0,
                                 device=feat0.device, dtype=feat0.dtype)],
                    dim=-2
                )
            elif mask_w0 != W0:
                feat0 = torch.cat(
                    [feat0,
                     torch.zeros(bs, c, H0, W0 - mask_w0,
                                 device=feat0.device, dtype=feat0.dtype)],
                    dim=-1
                )

            bs, c, mask_h1, mask_w1 = feat1.size()
            if mask_h1 != H1:
                feat1 = torch.cat(
                    [feat1,
                     torch.zeros(bs, c, H1 - mask_h1, W1,
                                 device=feat1.device, dtype=feat1.dtype)],
                    dim=-2
                )
            elif mask_w1 != W1:
                feat1 = torch.cat(
                    [feat1,
                     torch.zeros(bs, c, H1, W1 - mask_w1,
                                 device=feat1.device, dtype=feat1.dtype)],
                    dim=-1
                )

        return feat0, feat1
