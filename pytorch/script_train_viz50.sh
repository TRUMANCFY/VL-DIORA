#!/bin/bash
# sb --gres=gpu:titan_xp:rtx --cpus-per-task=16 --mem=100G coco_run.sh

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="8080"
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


srun python -m torch.distributed.launch diora/scripts/train_viz.py \
    --arch mlp-shared \
    --batch_size 16 \
    --data_type viz \
    --emb resnet50 \
    --hidden_dim 2048 \
    --log_every_batch 500 \
    --lr 1e-4 \
    --normalize unit \
    --reconstruct_mode softmax \
    --save_after 100 \
    --train_filter_length 20 \
    --train_path './data/partit_data/1.table/train' \
    --validation_path './data/partit_data/1.table/test' \
    --vision_type 'table' \
    --max_epoch 100 \
    --master_port 29502 \
    --word2idx './data/partit_data/partnet.dict.pkl' \
    --vocab_size 100 \
    --vision_pretrain_path '/itet-stor/fencai/net_scratch/VLGrammar/SCAN/outputs/partnet/table/scan/model-resnet50.pth.tar_64' \
    --freeze_model 1 \
    --save_distinct 500 \
    --cuda
    
echo finished at: `date`
exit 0;