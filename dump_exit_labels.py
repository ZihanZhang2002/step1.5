#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dump per-image-pair per-layer AUC@{5,10,20} for exit-gate (Stage-1.5).

It traverses a dataloader, and for each sample & each cross-block:
  - run coarse+fine matching at that block output
  - estimate relative pose (RANSAC / LO-RANSAC per config)
  - compute pose_err = max(R_err, t_err) in degrees
  - compute per-sample AUC@5/10/20 consistent with metrics.error_auc() when N=1
  - dump to a single npz

Usage example:

    python dump_exit_labels.py \
        configs/data/megadepth_trainval_832.py \
        configs/loftr/eloftr_full.py \
        --ckpt_path path/to/random_exit_teacher.ckpt \
        --dump_path dump/exit_perpair_perlayer_auc_megadepth_train.npz \
        --batch_size 4 \
        --num_workers 4 \
        --gpus 1
"""

import math
import argparse
from pathlib import Path
from distutils.util import strtobool

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
from src.utils.profiler import build_profiler

# reuse pose helpers from metrics.py (no need to modify metrics.py)
from src.utils.metrics import estimate_pose, estimate_lo_pose, relative_pose_error


loguru_logger = get_rank_zero_only_logger(logger)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")

    parser.add_argument("--exp_name", type=str, default="dump_exit_auc_per_pair_per_layer")
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
        "--disable_mp",
        action="store_true",
        help="disable mixed-precision",
    )
    parser.add_argument(
        "--deter",
        action="store_true",
        help="use deterministic mode for dumping",
    )

    parser.add_argument(
        "--dump_path",
        type=str,
        required=True,
        help="where to save the dumped npz",
    )

    # keep compatible with your datamodule/trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def single_sample_auc_from_pose_err(pose_err_deg: float, thr_deg: float) -> float:
    """
    Per-sample AUC consistent with metrics.error_auc() when errors list has only ONE element.
    If e >= thr -> 0
    Else -> 1 - 0.5 * e/thr
    """
    e = float(pose_err_deg)
    t = float(thr_deg)
    if not np.isfinite(e):
        return 0.0
    if e >= t:
        return 0.0
    return max(0.0, 1.0 - 0.5 * (e / t))


def estimate_pose_err_for_one_pair(kpts0, kpts1, K0, K1, T_0to1, config) -> float:
    """
    Return pose_err (deg) = max(R_err, t_err) for ONE image pair (one sample),
    using the same RANSAC method & threshold from config.
    Deterministic: no random shuffling, one call.
    """
    pixel_thr = float(config.TRAINER.RANSAC_PIXEL_THR)
    conf = float(config.TRAINER.RANSAC_CONF)
    method = str(config.TRAINER.POSE_ESTIMATION_METHOD)

    # too few matches => treat as failure
    if kpts0 is None or kpts1 is None or len(kpts0) < 5:
        return 90.0

    if method == "RANSAC":
        ret = estimate_pose(kpts0, kpts1, K0, K1, pixel_thr, conf=conf)
        if ret is None:
            return 90.0
        R, t, _inliers = ret
        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
        return float(max(t_err, R_err))

    elif method == "LO-RANSAC":
        est = estimate_lo_pose(kpts0, kpts1, K0, K1, pixel_thr, conf=conf)
        if not est["success"]:
            return 90.0
        M = est["M_0to1"]
        t_err, R_err = relative_pose_error(T_0to1, M.R, M.t, ignore_gt_t_thr=0.0)
        return float(max(t_err, R_err))

    else:
        raise ValueError(f"Unknown POSE_ESTIMATION_METHOD: {method}")


@torch.no_grad()
def main():
    args = parse_args()
    loguru_logger.info(f"Args: {vars(args)}")

    # ========== 1) load config ==========
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.main_cfg_path)
    cfg.merge_from_file(args.data_cfg_path)

    if cfg.LOFTR.COARSE.NPE is None:
        cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    if args.deter:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pl.seed_everything(cfg.TRAINER.SEED)

    # device
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    device = torch.device("cuda:0") if (_n_gpus > 0 and torch.cuda.is_available()) else torch.device("cpu")
    loguru_logger.info(f"Using device: {device}")

    # no training, but keep scaling fields valid
    cfg.TRAINER.WORLD_SIZE = 1
    cfg.TRAINER.TRUE_BATCH_SIZE = args.batch_size
    cfg.TRAINER.SCALING = cfg.TRAINER.TRUE_BATCH_SIZE / cfg.TRAINER.CANONICAL_BS
    cfg.TRAINER.TRUE_LR = cfg.TRAINER.CANONICAL_LR * cfg.TRAINER.SCALING
    cfg.TRAINER.WARMUP_STEP = math.floor(cfg.TRAINER.WARMUP_STEP / cfg.TRAINER.SCALING)

    # critical: we want all layer outputs
    cfg.LOFTR.COARSE.RANDOM_EXIT = False
    cfg.LOFTR.COARSE.SAVE_ALL_LAYERS = True

    # avoid mp for stability
    cfg.LOFTR.MP = False

    # ========== 2) build model ==========
    profiler = build_profiler(args.profiler_name)
    pl_model = PL_LoFTR(cfg, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    matcher = pl_model.matcher.to(device).eval()
    for p in matcher.parameters():
        p.requires_grad_(False)

    # identify cross blocks
    lft = matcher.loftr_coarse
    layer_names = lft.layer_names
    cross_layer_indices = [i for i, name in enumerate(layer_names) if name == "cross"]
    num_blocks = len(cross_layer_indices)
    loguru_logger.info(f"Cross layer indices: {cross_layer_indices}, num_blocks={num_blocks}")

    # ========== 3) dataloader ==========
    data_module = MultiSceneDataModule(args, cfg)
    data_module.setup(stage="fit")
    loader = data_module.train_dataloader()

    # ========== 4) dump ==========
    dump_path = Path(args.dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = [5.0, 10.0, 20.0]  # degrees

    identifiers_all = []
    pose_errs_all = []     # list of [num_blocks]
    aucs_all = []          # list of [num_blocks, 3]
    succs_all = []         # list of [num_blocks, 3]
    num_matches_all = []   # list of [num_blocks]

    backbone = matcher.backbone
    coarse_transformer = matcher.loftr_coarse
    coarse_matcher = matcher.coarse_matching
    fine_preprocess = matcher.fine_preprocess
    fine_matcher = matcher.fine_matching

    loguru_logger.info("Start dumping per-pair per-layer AUC@{5,10,20} ...")

    for batch in tqdm(loader, desc="Dumping", ncols=120):
        # to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        bs = batch["image0"].size(0)
        batch["bs"] = bs
        batch["hw0_i"] = batch["image0"].shape[2:]
        batch["hw1_i"] = batch["image1"].shape[2:]

        # -------- backbone --------
        if batch["hw0_i"] == batch["hw1_i"]:
            ret = backbone(torch.cat([batch["image0"], batch["image1"]], dim=0))
            feats_c = ret["feats_c"]
            batch.update({"feats_x2": ret["feats_x2"], "feats_x1": ret["feats_x1"]})
            feat_c0, feat_c1 = feats_c.split(bs)
        else:
            ret0 = backbone(batch["image0"])
            ret1 = backbone(batch["image1"])
            feat_c0, feat_c1 = ret0["feats_c"], ret1["feats_c"]
            batch.update(
                {
                    "feats_x2_0": ret0["feats_x2"],
                    "feats_x1_0": ret0["feats_x1"],
                    "feats_x2_1": ret1["feats_x2"],
                    "feats_x1_1": ret1["feats_x1"],
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

        mask_c0 = mask_c1 = None
        if "mask0" in batch:
            mask_c0, mask_c1 = batch["mask0"], batch["mask1"]

        # -------- coarse transformer (save all layers) --------
        _ = coarse_transformer(feat_c0, feat_c1, mask0=mask_c0, mask1=mask_c1, data=batch)
        feats0_all = batch.get("coarse_feats0_all", None)
        feats1_all = batch.get("coarse_feats1_all", None)
        if feats0_all is None or feats1_all is None:
            raise RuntimeError("SAVE_ALL_LAYERS=True but coarse_feats*_all not found in batch.")

        base_data = {
            k: v for k, v in batch.items()
            if k not in ["coarse_feats0_all", "coarse_feats1_all"]
        }

        mask_c0_flat = (mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else None)
        mask_c1_flat = (mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else None)

        # per-b init
        pose_errs_b = np.full((bs, num_blocks), 90.0, dtype=np.float32)
        aucs_b = np.zeros((bs, num_blocks, 3), dtype=np.float32)
        succs_b = np.zeros((bs, num_blocks, 3), dtype=np.bool_)
        nmatch_b = np.zeros((bs, num_blocks), dtype=np.int32)

        # -------- per block: run matching then compute pose_err per sample --------
        for block_idx, layer_idx in enumerate(cross_layer_indices):
            feat0_l = feats0_all[layer_idx]
            feat1_l = feats1_all[layer_idx]
            feat0_seq = rearrange(feat0_l, "n c h w -> n (h w) c")
            feat1_seq = rearrange(feat1_l, "n c h w -> n (h w) c")

            # copy a clean data dict for this block
            data_l = {}
            for k, v in base_data.items():
                if torch.is_tensor(v):
                    data_l[k] = v.clone()
                else:
                    data_l[k] = v

            coarse_matcher(feat0_seq, feat1_seq, data_l, mask_c0=mask_c0_flat, mask_c1=mask_c1_flat)

            feat0_norm = feat0_seq / math.sqrt(feat0_seq.shape[-1])
            feat1_norm = feat1_seq / math.sqrt(feat1_seq.shape[-1])

            feat_f0_unfold, feat_f1_unfold = fine_preprocess(feat0_norm, feat1_norm, data_l)
            fine_matcher(feat_f0_unfold, feat_f1_unfold, data_l)

            # now data_l should have mkpts0_f, mkpts1_f, m_bids
            m_bids = data_l.get("m_bids", None)
            mkpts0 = data_l.get("mkpts0_f", None)
            mkpts1 = data_l.get("mkpts1_f", None)

            if m_bids is None or mkpts0 is None or mkpts1 is None:
                # no matches produced
                continue

            m_bids_np = m_bids.detach().cpu().numpy()
            mkpts0_np = mkpts0.detach().cpu().numpy()
            mkpts1_np = mkpts1.detach().cpu().numpy()

            K0 = data_l["K0"].detach().cpu().numpy()
            K1 = data_l["K1"].detach().cpu().numpy()
            T_0to1 = data_l["T_0to1"].detach().cpu().numpy()

            for b in range(bs):
                mask = (m_bids_np == b)
                bpts0 = mkpts0_np[mask]
                bpts1 = mkpts1_np[mask]
                nmatch_b[b, block_idx] = int(mask.sum())

                e = estimate_pose_err_for_one_pair(
                    bpts0, bpts1,
                    K0[b], K1[b], T_0to1[b],
                    cfg
                )
                pose_errs_b[b, block_idx] = float(e)

                for ti, thr in enumerate(thresholds):
                    aucs_b[b, block_idx, ti] = single_sample_auc_from_pose_err(e, thr)
                    succs_b[b, block_idx, ti] = (e < thr)

        # -------- identifiers & append --------
        rel_pair_names = list(zip(*batch["pair_names"]))  # [(scene, pair_id), ...]
        for b in range(bs):
            identifiers_all.append("#".join(rel_pair_names[b]))
            pose_errs_all.append(pose_errs_b[b])
            aucs_all.append(aucs_b[b])
            succs_all.append(succs_b[b])
            num_matches_all.append(nmatch_b[b])

    # stack
    identifiers_all = np.array(identifiers_all)
    pose_errs_all = np.stack(pose_errs_all, axis=0).astype(np.float32)      # [N, B]
    aucs_all = np.stack(aucs_all, axis=0).astype(np.float32)               # [N, B, 3]
    succs_all = np.stack(succs_all, axis=0).astype(np.bool_)               # [N, B, 3]
    num_matches_all = np.stack(num_matches_all, axis=0).astype(np.int32)   # [N, B]

    np.savez_compressed(
        dump_path,
        identifiers=identifiers_all,
        pose_errs=pose_errs_all,
        aucs=aucs_all,
        succs=succs_all,
        num_matches=num_matches_all,
        cross_layer_indices=np.array(cross_layer_indices, dtype=np.int64),
        thresholds=np.array(thresholds, dtype=np.float32),
    )

    loguru_logger.info(f"Dumped {len(identifiers_all)} pairs to: {dump_path}")
    loguru_logger.info(f"pose_errs: {pose_errs_all.shape}, aucs: {aucs_all.shape}, succs: {succs_all.shape}")


if __name__ == "__main__":
    main()
