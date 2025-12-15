#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Measure per-exit latency (ms) for ELoFTR early-exit tradeoff.

Typical usage (MegaDepth test 1500):
    python scripts/profile/measure_exit_latency.py \
        configs/data/megadepth_test_1500.py \
        configs/loftr/eloftr_full.py \
        --ckpt_path path/to/ckpt.ckpt \
        --batch_size 1 \
        --num_workers 4 \
        --gpus 1 \
        --warmup 50 \
        --iters 300 \
        --out_npz dump/exit_latency_md1500.npz

Notes:
- If your coarse transformer supports *fixed exit*, we will measure real prefix latency per block.
- If not, we fallback to SAVE_ALL_LAYERS + per-layer matching-only timing (upper bound / not true prefix tradeoff).
"""

import os
import math
import json
import time
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
from src.lightning.lightning_loftr import PL_LoFTR, reparameter
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler


loguru_logger = get_rank_zero_only_logger(logger)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")

    parser.add_argument("--exp_name", type=str, default="measure_exit_latency")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=True,
        help="whether loading data to pinned memory or not",
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--profiler_name", type=str, default=None)
    parser.add_argument("--parallel_load_data", action="store_true")
    parser.add_argument("--disable_mp", action="store_true")
    parser.add_argument("--deter", action="store_true")

    parser.add_argument("--warmup", type=int, default=50, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=300, help="measured iterations")
    parser.add_argument("--max_batches", type=int, default=None, help="cap dataloader batches (debug)")

    parser.add_argument("--out_npz", type=str, default="dump/exit_latency.npz")
    parser.add_argument("--out_json", type=str, default=None)

    # what to time
    parser.add_argument(
        "--time_what",
        type=str,
        default="end2end",
        choices=["end2end", "matching_only"],
        help=(
            "end2end: backbone + coarse(prefix) + matching+fine. "
            "matching_only: only coarse_matching+fine_preprocess+fine_matching (requires SAVE_ALL_LAYERS)."
        ),
    )

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def _try_set_fixed_exit(matcher, block_idx: int) -> bool:
    """
    Try best-effort ways to force a fixed exit block.
    Return True if we believe the model will actually early-exit.
    """
    lft = matcher.loftr_coarse

    # 1) common attribute name patterns
    for attr in [
        "fixed_exit_block_idx",
        "fixed_exit_idx",
        "exit_block_idx",
        "force_exit_block_idx",
    ]:
        if hasattr(lft, attr):
            try:
                setattr(lft, attr, int(block_idx))
                return True
            except Exception:
                pass

    # 2) if lft has a setter
    for fn in ["set_fixed_exit", "set_exit_block", "set_exit_idx"]:
        if hasattr(lft, fn) and callable(getattr(lft, fn)):
            try:
                getattr(lft, fn)(int(block_idx))
                return True
            except Exception:
                pass

    # 3) sometimes stored in matcher.config / lft.config dict
    for obj in [matcher, lft]:
        if hasattr(obj, "config") and isinstance(obj.config, dict):
            # try common keys
            for key in [
                "fixed_exit_block_idx",
                "fixed_exit_idx",
                "exit_block_idx",
                "force_exit_block_idx",
            ]:
                try:
                    obj.config[key] = int(block_idx)
                    return True
                except Exception:
                    pass

    return False


@torch.no_grad()
def main():
    args = parse_args()
    loguru_logger.info(f"Args: {vars(args)}")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.main_cfg_path)
    cfg.merge_from_file(args.data_cfg_path)

    if cfg.LOFTR.COARSE.NPE is None:
        cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    if args.deter:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    pl.seed_everything(cfg.TRAINER.SEED)

    args.gpus = _n_gpus = setup_gpus(args.gpus)
    if _n_gpus > 0 and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    loguru_logger.info(f"Using device: {device}")

    # no training
    cfg.TRAINER.WORLD_SIZE = 1
    cfg.TRAINER.TRUE_BATCH_SIZE = args.batch_size
    cfg.TRAINER.SCALING = cfg.TRAINER.TRUE_BATCH_SIZE / cfg.TRAINER.CANONICAL_BS
    cfg.TRAINER.TRUE_LR = cfg.TRAINER.CANONICAL_LR * cfg.TRAINER.SCALING
    cfg.TRAINER.WARMUP_STEP = math.floor(cfg.TRAINER.WARMUP_STEP / cfg.TRAINER.SCALING)

    # latency test: no mp by default (you can change if you want)
    if args.disable_mp:
        cfg.LOFTR.MP = False

    # We may need SAVE_ALL_LAYERS for matching_only mode or fallback mode
    if args.time_what == "matching_only":
        cfg.LOFTR.COARSE.SAVE_ALL_LAYERS = True

    # turn off random-exit by default (we will force fixed exit if possible)
    cfg.LOFTR.COARSE.RANDOM_EXIT = False

    profiler = build_profiler(args.profiler_name)
    pl_model = PL_LoFTR(cfg, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    matcher = pl_model.matcher.to(device).eval()

    for p in matcher.parameters():
        p.requires_grad_(False)

    # RepVGG deploy
    if (cfg.LOFTR.BACKBONE_TYPE == "RepVGG"):
        try:
            matcher = reparameter(matcher)
            loguru_logger.info("RepVGG switched to deploy (reparameter).")
        except Exception as e:
            loguru_logger.warning(f"RepVGG reparameter failed (continue): {e}")

    # identify cross blocks
    lft = matcher.loftr_coarse
    layer_names = getattr(lft, "layer_names", None)
    if layer_names is None:
        raise RuntimeError("matcher.loftr_coarse.layer_names not found.")
    cross_layer_indices = [i for i, name in enumerate(layer_names) if name == "cross"]
    num_blocks = len(cross_layer_indices)
    loguru_logger.info(f"Coarse layer_names: {layer_names}")
    loguru_logger.info(f"Cross layer indices: {cross_layer_indices} (num_blocks={num_blocks})")

    # dataloader (use test loader by giving test cfg; MultiSceneDataModule will follow configs)
    data_module = MultiSceneDataModule(args, cfg)
    data_module.setup(stage="test")
    loader = data_module.test_dataloader()
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    # modules
    backbone = matcher.backbone
    coarse_transformer = matcher.loftr_coarse
    coarse_matcher = matcher.coarse_matching
    fine_preprocess = matcher.fine_preprocess
    fine_matcher = matcher.fine_matching

    # timing events
    use_cuda_timer = (device.type == "cuda")
    start_ev = torch.cuda.Event(enable_timing=True) if use_cuda_timer else None
    end_ev = torch.cuda.Event(enable_timing=True) if use_cuda_timer else None

    # book-keeping
    lat_ms = np.zeros((num_blocks,), dtype=np.float64)
    lat_ms2 = np.zeros((num_blocks,), dtype=np.float64)
    counts = np.zeros((num_blocks,), dtype=np.int64)

    # determine whether we have real fixed exit
    fixed_exit_supported = _try_set_fixed_exit(matcher, 0)
    if fixed_exit_supported:
        loguru_logger.info("Detected fixed-exit control (best-effort). Measuring REAL end2end prefix latency per block.")
    else:
        if args.time_what == "end2end":
            loguru_logger.warning(
                "Fixed-exit not detected. end2end timing will NOT reflect prefix savings unless your model truly early-exits.\n"
                "Consider implementing fixed-exit in LocalFeatureTransformer; otherwise use matching_only mode as a proxy."
            )
        else:
            loguru_logger.info("Fixed-exit not detected. Will run matching_only timing using SAVE_ALL_LAYERS.")

    # -------------------------
    # helper: run one forward for a given exit block
    # -------------------------
    def run_and_time_one_block(batch, block_idx: int):
        # move tensors
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        bs = batch["image0"].size(0)
        batch["bs"] = bs
        batch["hw0_i"] = batch["image0"].shape[2:]
        batch["hw1_i"] = batch["image1"].shape[2:]

        # coarse feature
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

        # for timing: sync
        if device.type == "cuda":
            torch.cuda.synchronize()

        # ---- start timing ----
        if use_cuda_timer:
            start_ev.record()
        else:
            t0 = time.perf_counter()

        if args.time_what == "end2end":
            # Try to force the model to exit at this block (if supported)
            _ = _try_set_fixed_exit(matcher, block_idx)

            # forward coarse transformer (should early-exit if fixed-exit is truly implemented)
            feat_c0_out, feat_c1_out = coarse_transformer(
                feat_c0, feat_c1, mask0=mask_c0, mask1=mask_c1, data=batch
            )

            # If SAVE_ALL_LAYERS is enabled, optionally take the exact layer output of this block
            # (useful if your coarse transformer doesn't actually early-exit)
            feats0_all = batch.get("coarse_feats0_all", None)
            feats1_all = batch.get("coarse_feats1_all", None)
            if feats0_all is not None and feats1_all is not None:
                layer_idx = cross_layer_indices[block_idx]
                feat0_l = feats0_all[layer_idx]
                feat1_l = feats1_all[layer_idx]
            else:
                feat0_l = feat_c0_out
                feat1_l = feat_c1_out

            feat0_seq = rearrange(feat0_l, "n c h w -> n (h w) c")
            feat1_seq = rearrange(feat1_l, "n c h w -> n (h w) c")

            mask_c0_flat = mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else None
            mask_c1_flat = mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else None

            coarse_matcher(feat0_seq, feat1_seq, batch, mask_c0=mask_c0_flat, mask_c1=mask_c1_flat)

            feat0_norm = feat0_seq / math.sqrt(feat0_seq.shape[-1])
            feat1_norm = feat1_seq / math.sqrt(feat1_seq.shape[-1])

            feat_f0_unfold, feat_f1_unfold = fine_preprocess(feat0_norm, feat1_norm, batch)
            fine_matcher(feat_f0_unfold, feat_f1_unfold, batch)

        else:
            # matching_only: need SAVE_ALL_LAYERS to get per-layer output
            cfg.LOFTR.COARSE.SAVE_ALL_LAYERS = True

            _ = coarse_transformer(
                feat_c0, feat_c1, mask0=mask_c0, mask1=mask_c1, data=batch
            )

            feats0_all = batch.get("coarse_feats0_all", None)
            feats1_all = batch.get("coarse_feats1_all", None)
            if feats0_all is None or feats1_all is None:
                raise RuntimeError("matching_only requires SAVE_ALL_LAYERS=True and coarse_feats*_all in batch.")

            layer_idx = cross_layer_indices[block_idx]
            feat0_l = feats0_all[layer_idx]
            feat1_l = feats1_all[layer_idx]

            feat0_seq = rearrange(feat0_l, "n c h w -> n (h w) c")
            feat1_seq = rearrange(feat1_l, "n c h w -> n (h w) c")

            mask_c0_flat = mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else None
            mask_c1_flat = mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else None

            coarse_matcher(feat0_seq, feat1_seq, batch, mask_c0=mask_c0_flat, mask_c1=mask_c1_flat)

            feat0_norm = feat0_seq / math.sqrt(feat0_seq.shape[-1])
            feat1_norm = feat1_seq / math.sqrt(feat1_seq.shape[-1])

            feat_f0_unfold, feat_f1_unfold = fine_preprocess(feat0_norm, feat1_norm, batch)
            fine_matcher(feat_f0_unfold, feat_f1_unfold, batch)

        # ---- end timing ----
        if use_cuda_timer:
            end_ev.record()
            torch.cuda.synchronize()
            ms = float(start_ev.elapsed_time(end_ev))
        else:
            ms = (time.perf_counter() - t0) * 1000.0

        return ms

    # -------------------------
    # warmup (use block 0)
    # -------------------------
    loguru_logger.info(f"Warmup {args.warmup} iters...")
    warm_loader = iter(loader)
    for _ in range(args.warmup):
        try:
            batch = next(warm_loader)
        except StopIteration:
            warm_loader = iter(loader)
            batch = next(warm_loader)
        _ = run_and_time_one_block(batch, block_idx=0)

    # -------------------------
    # measure
    # -------------------------
    loguru_logger.info(f"Measuring {args.iters} iters per block... time_what={args.time_what}")
    measure_loader = iter(loader)

    # We loop over blocks; for fairness, each block gets its own samples.
    for block_idx in range(num_blocks):
        pbar = tqdm(range(args.iters), desc=f"Block {block_idx}/{num_blocks-1}", ncols=110)
        for it in pbar:
            try:
                batch = next(measure_loader)
            except StopIteration:
                measure_loader = iter(loader)
                batch = next(measure_loader)

            ms = run_and_time_one_block(batch, block_idx=block_idx)

            # Welford-ish accumulate (mean/var)
            counts[block_idx] += 1
            lat_ms[block_idx] += ms
            lat_ms2[block_idx] += ms * ms

            if (it + 1) % 20 == 0:
                c = counts[block_idx]
                mean = lat_ms[block_idx] / max(c, 1)
                var = lat_ms2[block_idx] / max(c, 1) - mean * mean
                std = math.sqrt(max(var, 0.0))
                pbar.set_postfix(mean_ms=f"{mean:.2f}", std_ms=f"{std:.2f}")

            if args.max_batches is not None and it >= args.max_batches:
                break

    means = lat_ms / np.maximum(counts, 1)
    vars_ = lat_ms2 / np.maximum(counts, 1) - means * means
    stds = np.sqrt(np.maximum(vars_, 0.0))

    # output
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        means_ms=means.astype(np.float32),
        stds_ms=stds.astype(np.float32),
        counts=counts.astype(np.int64),
        cross_layer_indices=np.array(cross_layer_indices, dtype=np.int64),
        time_what=np.array([args.time_what]),
        upper_bound_mode=np.array([not fixed_exit_supported], dtype=np.bool_),
    )

    loguru_logger.info(f"Saved latency npz: {out_npz}")
    loguru_logger.info("Per-block latency (ms):")
    for i in range(num_blocks):
        loguru_logger.info(f"  block={i:02d} (layer_idx={cross_layer_indices[i]}): {means[i]:.3f} ± {stds[i]:.3f} ms")

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "means_ms": means.tolist(),
            "stds_ms": stds.tolist(),
            "counts": counts.tolist(),
            "cross_layer_indices": cross_layer_indices,
            "time_what": args.time_what,
            "upper_bound_mode": (not fixed_exit_supported),
        }
        out_json.write_text(json.dumps(payload, indent=2))
        loguru_logger.info(f"Saved latency json: {out_json}")


if __name__ == "__main__":
    main()
