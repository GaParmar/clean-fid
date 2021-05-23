#!/bin/bash

#################################################
# Get the relevant git repositoris
#################################################
if [ ! -d tmp/data-efficient-gans ]; then
    cd tmp
    git clone https://github.com/mit-han-lab/data-efficient-gans
    cd ..
fi
if [ ! -d tmp/stylegan2-ada-pytorch ]; then
    cd tmp
    git clone https://github.com/NVlabs/stylegan2-ada-pytorch
    cd ..
fi


#################################################
# Build Cifar-10 table
#################################################
for idx in 0 1 2 3 4 5 6; do
    args_str="--config_file CONFIGS.config_cifar_table --exp_idx $idx"
    args_str="$args_str --batch_size 50 --output_table OUT/table_cifar.json"
    ./parallelize_2gpu.sh $args_str --mode legacy_tensorflow
    # ./parallelize.sh $args_str --mode clean
done


#################################################
# Build Cifar-100 table
#################################################
# for idx in 0 1 2 3 4 5; do
#     args_str="--config_file CONFIGS.config_cifar100_table --exp_idx $idx"
#     args_str="$args_str --batch_size 32 --output_table OUT/table_cifar100.json"
#     ./parallelize.sh $args_str --mode legacy_tensorflow
#     ./parallelize.sh $args_str --mode clean
# done


#################################################
# LSUN - Cat (few shot)
#################################################
# for idx in 0 1 2 3 4 5 6 7; do
#     args_str="--config_file CONFIGS.config_lsuncat_table --exp_idx $idx"
#     args_str="$args_str --batch_size 32 --output_table OUT/table_lsuncat.json"
#     ./parallelize.sh $args_str --mode legacy_tensorflow
#     ./parallelize.sh $args_str --mode clean
# done


#################################################
# FFHQ - 256 (few shot)
#################################################
# for idx in 0 1 2 3 4 5; do
#     args_str="--config_file CONFIGS.config_ffhq256_table --exp_idx $idx"
#     args_str="$args_str --batch_size 32 --output_table OUT/table_ffhq256.json"
#     ./parallelize.sh $args_str --mode legacy_tensorflow
#     ./parallelize.sh $args_str --mode clean
# done


#################################################
# AFHQ - {Dog, Cat, Wild}
#################################################
# for idx in 0 1 2 3 4 5; do
#     args_str="--config_file CONFIGS.config_afhq --exp_idx $idx"
#     args_str="$args_str --batch_size 32 --output_table OUT/table_afhq.json"
#     ./parallelize.sh $args_str --mode legacy_tensorflow
#     ./parallelize.sh $args_str --mode legacy_tensorflow --metric "KID"
#     ./parallelize.sh $args_str --mode clean
#     ./parallelize.sh $args_str --mode clean --metric "KID"
# done

# #################################################
# # BreCaHAD
# #################################################
# for idx in 0 1; do
#     args_str="--config_file CONFIGS.config_brecahad --exp_idx $idx"
#     args_str="$args_str --batch_size 32 --output_table OUT/table_brecahad.json"
#     ./parallelize.sh $args_str --mode legacy_tensorflow
#     ./parallelize.sh $args_str --mode legacy_tensorflow --metric "KID"
#     ./parallelize.sh $args_str --mode clean
#     ./parallelize.sh $args_str --mode clean --metric "KID"
# done

#################################################
# MetFaces
#################################################
# for idx in 0 1; do
#     args_str="--config_file CONFIGS.config_metfaces --exp_idx $idx"
#     args_str="$args_str --batch_size 8 --output_table OUT/table_metfaces.json"
#     ./parallelize.sh $args_str --mode legacy_tensorflow
#     ./parallelize.sh $args_str --mode legacy_tensorflow --metric "KID"
#     ./parallelize.sh $args_str --mode clean
#     ./parallelize.sh $args_str --mode clean --metric "KID"
# done
