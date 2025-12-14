#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offline label dumping for exit-gate (Stage-1.5).

Usage example:

    python dump_exit_labels.py \
        configs/data/megadepth_trainval_832.py \
        configs/loftr/eloftr_full.py \
        --ckpt_path path/to/random_exit_teacher.ckpt \
        --dump_path dump/exit_labels_megadepth_train.npz \
        --batch_size 4 \
        --num_workers 4 \
        --gpus 1

它会遍历 train dataloader，
对每个样本、每个 cross-block 计算 coarse+fine 匹配，
用 match 数量决定 “best block”，输出到一个 npz 里。
"""

import os
import math
import argparse
from pathlib import Path
from distutils.util import strtobool  # ✅ 为了正确解析 --pin_memory

import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
from einops import rearrange

import pytorch_lightning as pl

from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR
from src.utils.misc import get_rank_zero_only_logger, setup_gpus

from src.utils.profiler import build_profiler  # ✅ 提前 import，避免在 main 里再 import


# 只在 rank0 打 log（虽然这里我们只用单进程，但保持一致风格）
loguru_logger = get_rank_zero_only_logger(logger)


def parse_args():
    """
    基本沿用 train.py 的 argparse，只是加了一个 --dump_path。
    这样 MultiSceneDataModule 可以直接复用，不用改 data 代码。
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 和 train.py 一样的前两个位置参数
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")

    # 下面这些也基本照抄 train.py，避免 DataModule 取不到字段
    parser.add_argument("--exp_name", type=str, default="dump_exit_labels")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=True,
        help="whether loading data to pinned memory or not",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Stage-1 random-exit teacher checkpoint path",
    )
    parser.add_argument(
        "--profiler_name",
        type=str,
        default=None,
        help="options: [inference, pytorch], or leave it unset",
    )
    parser.add_argument(
        "--parallel_load_data",
        action="store_true",
        help="load datasets with multiple processes.",
    )
    parser.add_argument(
        "--thr", type=float, default=0.1, help="coarse matching threshold override"
    )
    parser.add_argument(
        "--train_coarse_percent",
        type=float,
        default=0.1,
        help="(unused here, kept for compatibility)",
    )
    parser.add_argument(
        "--disable_mp",
        action="store_true",
        help="disable mixed-precision (我们本来就不用 mp)",
    )
    parser.add_argument(
        "--deter",
        action="store_true",
        help="use deterministic mode for dumping",
    )

    # 我们自己的额外参数：输出路径
    parser.add_argument(
        "--dump_path",
        type=str,
        required=True,
        help="where to save the dumped npz labels",
    )

    # 仍然加上 pl.Trainer 的 args，MultiSceneDataModule 不会用，但也不妨碍
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    loguru_logger.info(f"Args: {vars(args)}")

    # ========== 1. 读配置 & 修 config ==========
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.main_cfg_path)
    cfg.merge_from_file(args.data_cfg_path)

    # NPE 默认值（和 train.py 一样）
    if cfg.LOFTR.COARSE.NPE is None:
        cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    if args.deter:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pl.seed_everything(cfg.TRAINER.SEED)

    # GPU 设置：只用单进程，取第一个 GPU 即可
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    if _n_gpus > 0 and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    loguru_logger.info(f"Using device: {device}")

    # world_size 只影响 lr scaling，dump label 不训练，这里简单设成 1
    cfg.TRAINER.WORLD_SIZE = 1
    cfg.TRAINER.TRUE_BATCH_SIZE = args.batch_size
    cfg.TRAINER.SCALING = cfg.TRAINER.TRUE_BATCH_SIZE / cfg.TRAINER.CANONICAL_BS
    cfg.TRAINER.TRUE_LR = cfg.TRAINER.CANONICAL_LR * cfg.TRAINER.SCALING
    cfg.TRAINER.WARMUP_STEP = math.floor(
        cfg.TRAINER.WARMUP_STEP / cfg.TRAINER.SCALING
    )

    # 覆盖 coarse config：❗关掉 random-exit，打开 SAVE_ALL_LAYERS（为了拿全层特征）
    cfg.LOFTR.COARSE.RANDOM_EXIT = False
    cfg.LOFTR.COARSE.SAVE_ALL_LAYERS = True

    # eval 阶段不需要 MP，统一关掉，避免奇怪的数值问题
    cfg.LOFTR.MP = False

    # ========== 2. 构建 Lightning 模型（只用里面的 matcher） ==========
    profiler = build_profiler(args.profiler_name)
    pl_model = PL_LoFTR(cfg, pretrained_ckpt=args.ckpt_path, profiler=profiler)

    matcher = pl_model.matcher.to(device).eval()

    # ✅ 冻结所有参数（概念上更干净，虽然 @no_grad 已经保证不会算梯度）
    for p in matcher.parameters():
        p.requires_grad_(False)

    # 从 matcher 里拿到 coarse transformer 的结构，用来识别 cross-block
    lft = matcher.loftr_coarse
    layer_names = lft.layer_names
    cross_layer_indices = [i for i, name in enumerate(layer_names) if name == "cross"]
    num_blocks = len(cross_layer_indices)
    loguru_logger.info(f"Coarse layer_names: {layer_names}")
    loguru_logger.info(
        f"Cross layer indices (block ends): {cross_layer_indices}, num_blocks={num_blocks}"
    )

    # ========== 3. 构建 DataModule，只用 train_dataloader ==========
    data_module = MultiSceneDataModule(args, cfg)
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()

    # ========== 4. 遍历 train 数据，按 sample 生成 label ==========
    identifiers_all = []
    labels_all = []
    scores_all = []

    dump_path = Path(args.dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    matcher_backbone = matcher.backbone
    coarse_transformer = matcher.loftr_coarse
    coarse_matcher = matcher.coarse_matching
    fine_preprocess = matcher.fine_preprocess
    fine_matcher = matcher.fine_matching

    loguru_logger.info("Start dumping exit labels over train set...")

    for batch in tqdm(train_loader, desc="Dumping labels", ncols=120):
        # 把 tensor 丢到 device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        bs = batch["image0"].size(0)
        batch["bs"] = bs
        batch["hw0_i"] = batch["image0"].shape[2:]
        batch["hw1_i"] = batch["image1"].shape[2:]

        # ====== (1) Backbone: coarse feature ======
        if batch["hw0_i"] == batch["hw1_i"]:
            ret_dict = matcher_backbone(
                torch.cat([batch["image0"], batch["image1"]], dim=0)
            )
            feats_c = ret_dict["feats_c"]
            batch.update(
                {
                    "feats_x2": ret_dict["feats_x2"],
                    "feats_x1": ret_dict["feats_x1"],
                }
            )
            feat_c0, feat_c1 = feats_c.split(bs)
        else:
            ret_dict0 = matcher_backbone(batch["image0"])
            ret_dict1 = matcher_backbone(batch["image1"])
            feat_c0 = ret_dict0["feats_c"]
            feat_c1 = ret_dict1["feats_c"]
            batch.update(
                {
                    "feats_x2_0": ret_dict0["feats_x2"],
                    "feats_x1_0": ret_dict0["feats_x1"],
                    "feats_x2_1": ret_dict1["feats_x2"],
                    "feats_x1_1": ret_dict1["feats_x1"],
                }
            )

        mul = matcher.config["resolution"][0] // matcher.config["resolution"][1]
        batch.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
                "hw1_f": [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
            }
        )

        # coarse-level mask
        mask_c0 = mask_c1 = None
        if "mask0" in batch:
            mask_c0, mask_c1 = batch["mask0"], batch["mask1"]

        # ====== (2) Coarse transformer: 保存所有层输出 ======
        feat_c0_trans, feat_c1_trans = coarse_transformer(
            feat_c0, feat_c1, mask0=mask_c0, mask1=mask_c1, data=batch
        )
        # batch 里此时多了：
        #   batch['coarse_feats0_all']: list[len = num_layers] of [B, C, Hc, Wc]
        #   batch['coarse_feats1_all']: 同上
        feats0_all = batch.get("coarse_feats0_all", None)
        feats1_all = batch.get("coarse_feats1_all", None)
        if feats0_all is None or feats1_all is None:
            raise RuntimeError(
                "SAVE_ALL_LAYERS should be True, but coarse_feats*_all not found in batch."
            )

        # 之后的 coarse/fine/matching 我们要自己按 “block” 重复跑多次，
        # 所以先构一个 base_data，避免在同一个 dict 上反复 in-place 改得太乱。
        base_data = {
            k: v
            for k, v in batch.items()
            if k not in ["coarse_feats0_all", "coarse_feats1_all"]
        }

        # ====== (3) 对每一个 cross-block 都跑一遍 matching，统计 score ======
        scores = torch.zeros(bs, num_blocks, device=device)

        # flatten mask for coarse_matching
        mask_c0_flat = (
            mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else None
        )
        mask_c1_flat = (
            mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else None
        )

        for block_idx, layer_idx in enumerate(cross_layer_indices):
            feat0_l = feats0_all[layer_idx]  # [B, C, Hc, Wc]
            feat1_l = feats1_all[layer_idx]

            feat0_seq = rearrange(feat0_l, "n c h w -> n (h w) c")
            feat1_seq = rearrange(feat1_l, "n c h w -> n (h w) c")

            # 为当前 block 复制一份 data，避免不同 block 之间互相污染
            data_l = {}
            for k, v in base_data.items():
                if torch.is_tensor(v):
                    # 不一定必须 clone，但 clone 一下更安全
                    data_l[k] = v.clone()
                else:
                    data_l[k] = v

            # coarse matching
            coarse_matcher(
                feat0_seq,
                feat1_seq,
                data_l,
                mask_c0=mask_c0_flat,
                mask_c1=mask_c1_flat,
            )

            # 防止 fp16 溢出（这里本来就是 fp32，但保持一致）
            feat0_norm, feat1_norm = map(
                lambda feat: feat / math.sqrt(feat.shape[-1]),
                [feat0_seq, feat1_seq],
            )

            # fine-level refinement
            feat_f0_unfold, feat_f1_unfold = fine_preprocess(
                feat0_norm, feat1_norm, data_l
            )
            fine_matcher(feat_f0_unfold, feat_f1_unfold, data_l)

            # 质量评分：这里用 “每个样本的 match 数量” 作为 score
            # data_l['m_bids']: [N_matches], 表示每个 match 属于哪个 batch 内样本
            m_bids = data_l["m_bids"]  # tensor, shape [N_matches]
            for b in range(bs):
                scores[b, block_idx] = (m_bids == b).sum()

        # ====== (4) 为当前 batch 里的每个样本选 best block ======
        labels_batch = torch.argmax(scores, dim=1).cpu().numpy()  # [B]

        # 构建 identifier（和原 Lightning 里 metrics 一样）
        rel_pair_names = list(zip(*batch["pair_names"]))  # [(scene, pair_id), ...]
        for b in range(bs):
            identifier = "#".join(rel_pair_names[b])
            identifiers_all.append(identifier)
            labels_all.append(int(labels_batch[b]))
            scores_all.append(scores[b].cpu().numpy())

    # ========== 5. 全部 dump 到 npz ==========
    identifiers_all = np.array(identifiers_all)
    labels_all = np.array(labels_all, dtype=np.int64)
    scores_all = np.stack(scores_all, axis=0).astype(np.float32)
    cross_layer_indices_np = np.array(cross_layer_indices, dtype=np.int64)

    np.savez_compressed(
        dump_path,
        identifiers=identifiers_all,
        labels=labels_all,
        scores=scores_all,
        cross_layer_indices=cross_layer_indices_np,
    )

    loguru_logger.info(f"Dumped {len(labels_all)} samples to: {dump_path}")
    loguru_logger.info(
        f"labels shape: {labels_all.shape}, scores shape: {scores_all.shape}, "
        f"num_blocks={num_blocks}, cross_layer_indices={cross_layer_indices}"
    )


if __name__ == "__main__":
    main()
