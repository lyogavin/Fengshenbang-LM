#!/bin/bash
#SBATCH --job-name=pretrain_randeng_t5_char_700M
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o /cognitive_comp/ganruyi/experiments/randeng_t5_char_700M/%x-%j.log
#SBATCH -e /cognitive_comp/ganruyi/experiments/randeng_t5_char_700M/%x-%j.err

set -x -e

run_ts=$(date +%s)
echo "RUN TS: $(run_ts)"

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=4
ROOT_DIR=/home/ubuntu/cloudfs/saved_models/deep_speed_experiments/t5/
if [ ! -d ${ROOT_DIR} ];then
  mkdir -p ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

ZERO_STAGE=1

config_json="$ROOT_DIR/ds_config.randeng_t5_char_700M.$SLURM_JOBID.json"
export MASTER_PORT=$[RANDOM%10000+30000]
checkpoint_path=$ROOT_DIR/ckpt_$run_ts

echo "checkpoint @ $checkpoint_path"
# export CUDA_VISIBLE_DEVICES='2,5'

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 100,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "contiguous_gradients": false,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 500000000
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-4,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "params": {
      "warmup_max_lr": 1e-04,
      "warmup_min_lr": 1e-05,
      "total_num_steps": 400000,
      "warmup_num_steps" : 10000
    },
    "type": "WarmupDecayLR"  
  },
  "zero_allow_untested_optimizer": false,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json

# https://pytorch.org/docs/stable/cpp_extension.html
# use default
#export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions

# strategy=ddp
strategy=deepspeed_stage_1

    #--dirpath $ROOT_DIR/ckpt \

# val_check_interval 0.25 check validation set 4 times during a training epoch
# every_n_train_steps (Optional[int]) – Number of training steps between checkpoints.
TRAINER_ARGS="
    --max_epochs 1 \
    --gpus 1 \
    --num_nodes 1 \
    --strategy ${strategy} \
    --default_root_dir $ROOT_DIR \
    --save_top_k 3 \
    --every_n_train_steps 100000 \
    --monitor train_loss \
    --mode min \
    --save_last \
    --val_check_interval 0.5 \
    --dataset_num_workers 4 \
    --dataloader_num_workers 4 \
    --replace_sampler_ddp False \
    --accumulate_grad_batches 2 \
    --save_ckpt_path $checkpoint_path \
"
# --accumulate_grad_batches 8 \
# sample test
#DATA_DIR=/home/ubuntu/cloudfs/ghost_data/merge_all_title_content/merged_all_title_content_1017_1665986569_sample_test.csv.gz # for test
#VAL_DATA_DIR=/home/ubuntu/cloudfs/ghost_data/merge_all_title_content/merged_all_title_content_1017_1665986569_sample_test.csv.gz # for test
# real train
DATA_DIR=/home/ubuntu/cloudfs/ghost_data/merge_all_title_content/merged_all_title_content_1017_train_1665986569.csv.gz
VAL_DATA_DIR=/home/ubuntu/cloudfs/ghost_data/merge_all_title_content/merged_all_title_content_1017_val_1665986569.csv.gz

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --text_column_name title_content
    --train_data_path ${DATA_DIR} \
    --val_data_path ${VAL_DATA_DIR} \
    --train_split_size 0.999 \
    --max_seq_length 512 \
    --overwrite_cache \
    --dataset_name all_title_content_1017 \
"

MODEL_ARGS="
    --pretrained_model_path IDEA-CCNL/Randeng-T5-784M \
    --tokenizer_type t5_tokenizer \
"
#bert_tokenizer


SCRIPTS_PATH=/home/ubuntu/Fengshenbang-LM/fengshen/examples/pretrain_t5/pretrain_t5.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
# /home/ganruyi/anaconda3/bin/python $CMD
#SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif
#srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c '/home/ganruyi/anaconda3/bin/python $CMD'
python $CMD

# source activate base
# python $CMD
# srun --nodes=1 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=30 --jobid=171866 -e %x-%j.err -o %x-%j.log python $CMD

