#!/usr/bin/env bash

log_folder="output"
gpu=2
model_name='RoadEA'

while getopts "g:l:" opt;
do
    case ${opt} in
        g) gpu=$OPTARG ;;
        l) log_folder=$OPTARG ;;
        *) ;;
    esac
done

echo "log folder: " "${log_folder}"
if [[ ! -d ${log_folder} ]];then
    mkdir -p "${log_folder}"
    echo "create log folder: " "${log_folder}"
fi

datasets=('DBP_en_DBP_fr_15K')
dim=(320)
learning_rate=(0.0001)
attr_len=(10)
rel_len=(25)
mapping_batch_size=(1024)
neg_pro=(0.96)
input_drop_rate=(0.2)

prefix='../datasets/'
suffix='/'

for ds in "${datasets[@]}"
do
for dm in "${dim[@]}"
do
for lr in "${learning_rate[@]}"
do
for al in "${attr_len[@]}"
do
for rl in "${rel_len[@]}"
do
for bs in "${mapping_batch_size[@]}"
do
for np in "${neg_pro[@]}"
do
for idr in "${input_drop_rate[@]}"
do
path=${prefix}${datasets}${suffix}

cur_time="$(date +%Y%m%d%H%M%S)"
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                              --model "${model_name}" \
                              --input "${path}" \
                              --dim "${dm}" \
                              --learning_rate "${lr}" \
                              --attr_len "${al}" \
                              --rel_len "${rl}" \
                              --neg_pro "${np}" \
                              --mapping_batch_size "${bs}" \
                              > "${log_folder}"/"${model_name}"_"${ds}"_"${al}"_lr"${lr}"_neg_pro"${np}"_bs"${bs}"_"${cur_time}"
sleep 10
done
done
done
done
done
done
done
done
