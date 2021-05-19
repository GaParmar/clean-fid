#!/bin/bash

for GPU_ID in 0 1 2 3
do
    seed=$(($GPU_ID + 0))
    CUDA_VISIBLE_DEVICES=$GPU_ID python base.py $* --seed $seed &
    pids[${GPU_ID}]=$!
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

for GPU_ID in 0 1 2 3
do
    seed=$(($GPU_ID + 4))
    CUDA_VISIBLE_DEVICES=$GPU_ID python base.py $* --seed $seed &
    pids[${GPU_ID}]=$!
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

for GPU_ID in 0 1
do
    seed=$(($GPU_ID + 8))
    CUDA_VISIBLE_DEVICES=$GPU_ID python base.py $* --seed $seed &
    pids[${GPU_ID}]=$!
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done