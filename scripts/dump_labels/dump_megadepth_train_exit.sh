#!/bin/bash -l
# Offline dump exit labels for MegaDepth train set (Stage-1.5)

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

########################
# Config & paths
########################
data_cfg_path="configs/data/megadepth_trainval_832.py"
main_cfg_path="configs/loftr/eloftr_full.py"

# ✅ 这里改成你训练好的 Stage-1 random-exit teacher 的 ckpt 路径
ckpt_path="weights/random_exit_teacher.ckpt"

# ✅ 输出的标签文件路径
dump_path="dump/exit_labels_megadepth_train.npz"

########################
# Hardware & dataloader
########################
n_nodes=1
n_gpus_per_node=1   # 单卡就够了
torch_num_workers=4
batch_size=4

########################
# Run
########################
python dump_exit_labels.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_path=${dump_path} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} \
    --batch_size=${batch_size} \
    --num_workers=${torch_num_workers} \
    --thr 0.1 \
    --deter
