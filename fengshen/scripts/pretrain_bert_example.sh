#!/bin/bash
#SBATCH --job-name=hf-bert-base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err

#SBATCH --qos=preemptive # specify preemptive QOS
#SBATCH -t 00:10:00

set -x -e

echo "START TIME: $(date)"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=127.0.0.1
MASTER_PORT=61000

GPUS_PER_NODE=2
NNODES=1   # switch to 128
TP_SIZE=1    # always fixed to the size of a single node
PP_SIZE=1    # NLAYERS must be a multiple of PP_SIZE here

MICRO_BATCH_SIZE=16

ZERO_STAGE=3

config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

TRAINER_ARGS="
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --output_dir /cognitive_comp/gaoxinyu/hf_model/bert-base/ \
    --fp16 True \
"

CNNL_TRAINER_ARGS="
    --megatron_data_path /cognitive_comp/gaoxinyu/data/WuDaoCorpus280G_split_pretrain/WuDao280G_splitaa_content_sentence \
"

DEEPSPEED_ARGS=" \
    --deepspeed ${config_json} \
    "

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    "

SCRIPTS_PATH=/cognitive_comp/gaoxinyu/FS/Fengshenbang-LM

export CMD=" \
    $SCRIPTS_PATH/pretrain_bert.py \
    $TRAINER_ARGS \
    $CNNL_TRAINER_ARGS \
    $DEEPSPEED_ARGS \
    "

echo $CMD

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH $LAUNCHER --node_rank 0 $CMD

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
#clear; srun --jobid $SLURM_JOBID singularity exec --nv -B /cognitive_comp/ganruyi/Megatron/:/cognitive_comp/ganruyi/Megatron/,/cognitive_comp/gaoxinyu/:/cognitive_comp/gaoxinyu/ $SINGULARITY_PATH bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'