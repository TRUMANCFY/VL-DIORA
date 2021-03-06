#!/bin/bash
# sb --gres=gpu:titan_xp:rtx --cpus-per-task=16 --mem=100G coco_run.sh

export MASTER_ADDR="0.0.0.0"
export MASTER_PORT="8088"
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

# srun python diora/scripts/parse.py \
#     --batch_size 10 \
#     --data_type txt_id \
#     --elmo_cache_dir data/elmo \
#     --load_model_path ../Downloads/diora-checkpoints/mlp-softmax-shared/model.pt \
#     --model_flags ../Downloads/diora-checkpoints/mlp-softmax-shared/flags.json \
#     --validation_path ./sample.txt \
#     --validation_filter_length 10

# 1e1044b6
# model.step_700.pt

# ec6b79bd

# d6152b49
# model.step_900.pt

# 437752: be50a128

# bed: fc14bc07

# table: 26074f4a model.step_2700.pt

# chair: 3fdb81a3 model.step_2500.pt

# bag: 3b010f77 mode.step_1400.pt

srun python diora/scripts/parse_bert.py \
    --batch_size 10 \
    --data_type partit \
    --emb bert \
    --load_model_path ../log/3b010f77/model.step_1400.pt \
    --model_flags ../log/3b010f77/flags.json \
    --validation_path ./data/partit_data/3.bag/test \
    --validation_filter_length 20 \
    --word2idx './data/partit_data/partnet.dict.pkl' \
    --k_neg 5 \
    --freeze_bert 1 \
    --cuda

    # --elmo_cache_dir data/elmo \

echo finished at: `date`
exit 0;