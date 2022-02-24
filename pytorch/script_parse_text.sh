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

# 019e53de
# model.step_43000.pt
# model.step_52200.pt
# model.step_62300.pt

# cd9a1633
# model.step_173900.pt

# 992b3f0a
# model.step_173900.json

# 12.24 new
# b6705305
# model.step_5100.pt

# 598ee336

srun python diora/scripts/parse.py \
    --batch_size 10 \
    --data_type partit \
    --elmo_cache_dir data/elmo \
    --load_model_path ../log/b482bb14/model.step_4900.pt \
    --model_flags ../log/b482bb14/flags.json \
    --validation_path ./data/partit_data/1.table/test \
    --validation_filter_length 20 \
    --word2idx './data/partit_data/partnet.dict.pkl'

echo finished at: `date`
exit 0;