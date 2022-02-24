#!/bin/bash
# sb --gres=gpu:titan_xp:rtx --cpus-per-task=16 --mem=100G coco_run.sh

export MASTER_ADDR="127.0.0.1"
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
# export CUDA_VISIBLE_DEVICE=0,1
# export NGPUS=2


srun python -m torch.distributed.launch --nproc_per_node=1 diora/scripts/train_combine.py \
    --arch mlp-shared \
    --batch_size 8 \
    --data_type partit \
    --elmo_cache_dir ./data/elmo \
    --emb combine \
    --hidden_dim 512 \
    --k_neg 100 \
    --log_every_batch 300 \
    --reconstruct_mode softmax \
    --lr 2e-3 \
    --normalize unit \
    --save_after 500 \
    --cuda \
    --max_epoch 500 \
    --master_port 29500 \
    --word2idx './data/partit_data/partnet.dict.pkl' \
    --freeze_model 1 \
    --level_attn 1 \
    --diora_shared 0 \
    --mixture 1 \
    --txt2img 0 \
    --outside_attn 0 \
    --train_path './data/partit_data/3.bag/train' \
    --validation_path './data/partit_data/3.bag/test' \
    --vision_pretrain_path '/itet-stor/fencai/net_scratch/VLGrammar/SCAN/outputs/partnet/bag/scan/model-resnet18.pth.tar_90' \
    --vision_type 'bag'



    # 0.chair
    # --train_path './data/partit_data/0.chair/train' \
    # --validation_path './data/partit_data/0.chair/test' \
    # --vision_pretrain_path '/itet-stor/fencai/net_scratch/VLGrammar/SCAN/outputs/partnet/chair/scan/model-resnet18.pth.tar_40' \
    # --vision_type 'chair'
    # 1.table
    # --train_path './data/partit_data/1.table/train' \
    # --validation_path './data/partit_data/1.table/test' \
    # --vision_pretrain_path '/itet-stor/fencai/net_scratch/VLGrammar/SCAN/outputs/partnet/table/scan/model-resnet18.pth.tar_45' \
    # --vision_type 'table'
    # 3.bag
    # --train_path './data/partit_data/3.bag/train' \
    # --validation_path './data/partit_data/3.bag/test' \
    # --vision_pretrain_path '/itet-stor/fencai/net_scratch/VLGrammar/SCAN/outputs/partnet/bag/scan/model-resnet18.pth.tar_90' \
    # --vision_type 'bag'

echo finished at: `date`
exit 0;