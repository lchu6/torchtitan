#!/bin/bash

export job_name=lchu-titan-async
export num_nodes=8

bsub \
        -J $job_name \
        -gpu \"num=8/task:mode=exclusive_process\" \
        -n $num_nodes \
        -R "span[ptile=1]" \
        -gpu "num=8:j_exclusive=yes" \
        -o $job_name.out \
        blaunch ./train.sh
