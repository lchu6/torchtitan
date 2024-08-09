#!/bin/bash

export HF_HOME=/proj/data-eng/lchu/huggingface 

MODEL_ARGS="\
--job.config_file ./train_configs/llama3_70b.toml
"

torchrun \
    --nnodes=$num_nodes \
    --node_rank=`echo $(($LSF_PM_TASKID - 1))`  \
    --nproc_per_node=8  \
    --master-addr=`echo $LSB_HOSTS | xargs -n 1 | head -n 1` \
    --master-port="12234" \
    train.py $MODEL_ARGS
