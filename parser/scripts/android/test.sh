#!/bin/bash
model_name=$(basename $1)

parser='pretrain_seq2seq_ui'
evaluator="default_evaluator"
align_att=60
batch_size=128
data_dir_prefix="datasets/android/"
data_dir=${data_dir_prefix}
test_file=${data_dir}test.bin
python schema_sup_exp.py \
    --use_cuda \
    --mode test \
    --lang top \
    --load_model $1 \
    --beam_size 5 \
    --copy \
    --parser ${parser} \
    --relax_factor 10 \
    --batch_size ${batch_size} \
    --evaluator ${evaluator} \
    --test_file ${test_file} \
    --clip_grad_mode norm \
    --align_att ${align_att} \
    --decode_max_time_step 200 \
    #--save_decode_to ${decode_file_path}
    #--batch_decode \