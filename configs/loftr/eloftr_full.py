from src.config.default import _CN as cfg

# training config（其实 dump label 不训练，写不写关系不大，但保留原始设置没坏处）
cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5
cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.TRAINER.EPI_ERR_THR = 5e-4
cfg.TRAINER.GRADIENT_CLIPPING = 0.0
cfg.LOFTR.LOSS.FINE_TYPE = 'l2'
cfg.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = True
cfg.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = True
cfg.LOFTR.LOSS.LOCAL_WEIGHT = 0.25
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = True

# model config（保持和 Stage-1 结构一致）
cfg.LOFTR.RESOLUTION = (8, 1)
cfg.LOFTR.FINE_WINDOW_SIZE = 8
cfg.LOFTR.ALIGN_CORNER = False
cfg.LOFTR.MP = True
cfg.LOFTR.REPLACE_NAN = True
cfg.LOFTR.EVAL_TIMES = 5
cfg.LOFTR.COARSE.NO_FLASH = True
cfg.LOFTR.MATCH_COARSE.THR = 0.2
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8

# ⚠️ 关键：离线打标签时，不再 random-exit，而是变成“完整 8 层 teacher”
cfg.LOFTR.COARSE.RANDOM_EXIT = False        # 关掉 random-exit
cfg.LOFTR.COARSE.EXIT_ON_CROSS_ONLY = True  # 留着无所谓，不会用到
cfg.LOFTR.COARSE.EXIT_MIN_BLOCK = 0
cfg.LOFTR.COARSE.EXIT_MAX_BLOCK = 3

# 如果你准备一次 forward 存所有层，用于后续分析 / 更复杂的标签，可以开这个
# 否则就按“跑多次，每次强制不同 exit_block”来打标签
cfg.LOFTR.COARSE.SAVE_ALL_LAYERS = False  # 或 True，看你后面的打标签方案

# dataset config
cfg.DATASET.FP16 = False

# full model config
cfg.LOFTR.MATCH_COARSE.FP16MATMUL = False
