#!/bin/bash

export job_name=lchu-titan-fp8
export num_nodes=2

export HF_HOME=/proj/data-eng/lchu/huggingface 

export MODEL_ARGS="\
--job.config_file ./train_configs/llama3_8b.toml
"

TORCHRUN_COMMAND='\
torchrun \
    --nnodes=$num_nodes \
    --node_rank=`echo $(($LSF_PM_TASKID - 1))`  \
    --nproc_per_node=8  \
    --master-addr=`echo $LSB_HOSTS | xargs -n 1 | head -n 1` \
    --master-port=12234 \
    train.py $MODEL_ARGS
'

bsub \
        -J $job_name \
        -gpu \"num=8/task:mode=exclusive_process\" \
        -n $num_nodes \
        -R "span[ptile=1]" \
        -gpu "num=8:j_exclusive=yes" \
        -o $job_name.out \
        blaunch bash -c "$TORCHRUN_COMMAND"
