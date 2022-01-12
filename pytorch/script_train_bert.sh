#!/bin/bash
# sb --gres=gpu:titan_xp:rtx --cpus-per-task=16 --mem=100G coco_run.sh

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="8098"
export NODE_RANK=0

#SBATCH  --mail-type=ALL                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --gres=gpu:geforce_rtx_3090:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_IdD
#
# binary to execute
set -o errexit
source /itet-stor/fencai/net_scratch/anaconda3/bin/activate diora
export PYTHONPATH=/itet-stor/fencai/net_scratch/diora/pytorch/:$PYTHONPATH
# export CUDA_VISIBLE_DEVICE=0,1
# export NGPUS=2


srun python -m torch.distributed.launch --nproc_per_node=1 diora/scripts/train_bert.py \
    --arch mlp-shared \
    --batch_size 8 \
    --data_type partit \
    --emb bert \
    --hidden_dim 400 \
    --k_neg 30 \
    --log_every_batch 100 \
    --lr 1e-4 \
    --normalize unit \
    --reconstruct_mode softmax \
    --save_after 500 \
    --train_filter_length 20 \
    --train_path './data/partit_data/0.chair/train' \
    --validation_path './data/partit_data/0.chair/test' \
    --cuda \
    --max_epoch 200 \
    --master_port 29501 \
    --word2idx './data/partit_data/partnet.dict.pkl' \
    --freeze_bert 1

# srun python -m torch.distributed.launch --nproc_per_node=1 diora/scripts/train_bert.py \
#     --arch mlp-shared \
#     --batch_size 8 \
#     --data_type partitwhole \
#     --emb bert \
#     --hidden_dim 400 \
#     --k_neg 50 \
#     --log_every_batch 100 \
#     --lr 1e-4 \
#     --normalize unit \
#     --reconstruct_mode softmax \
#     --save_after 500 \
#     --train_filter_length 20 \
#     --train_path './data/partit_data/{}/train' \
#     --validation_path './data/partit_data/{}/test' \
#     --cuda \
#     --max_epoch 200 \
#     --master_port 29501 \
#     --word2idx './data/partit_data/partnet.dict.pkl' \
#     --freeze_bert 1

    # --tokenizer_loading_path './data/bert/toks/' \
    # --bertmodel_loading_path './data/bert/model/'
    # # --local_rank 0

echo finished at: `date`
exit 0;