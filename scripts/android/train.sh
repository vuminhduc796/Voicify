#!/bin/bash

set -e

data_dir_prefix="datasets/android_user/"
model_dir_prefix="saved_models/android_user/"
augment=0
vocab=${data_dir_prefix}"vocab.freq2.bin"
train_file=${data_dir_prefix}"train.bin"
#dev_file=${data_dir_prefix}"geo_en_prolog_test.bin"
model_dir=${model_dir_prefix}
align_att=60
dropout=0.3
hidden_size=768
decoder_embed_size=256
lr_decay=0.985
lr_decay_after_epoch=40
max_epoch=100
patience=1000   # disable patience since we don't have dev set
beam_size=1
batch_size=64
decoder_layer_size=2
valid_every_epoch=10
lr=0.001
ls=0
plmm_model_name='xlm-roberta-base'
attention='dot'
clip_grad_mode='norm'
lstm='lstm'
optimizer='Adam'
parser='pretrain_seq2seq_ui'
suffix='android_user'
model_name=model.android_eval.sup.${lstm}.hid${hidden_size}.de_embed${decoder_embed_size}.drop${dropout}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.ls${ls}.seed${seed}.clip_grad_mode${clip_grad_mode}.parser${parser}.suffix${suffix}

python3 -u schema_sup_exp.py \
    --use_cuda \
    --seed 0 \
    --mode train \
    --augment ${augment} \
    --lang top \
    --batch_size ${batch_size} \
    --copy \
    --use_adapter \
    --train_file ${train_file} \
    --vocab ${vocab} \
    --plmm_model_name ${plmm_model_name} \
    --align_att ${align_att} \
    --parser ${parser} \
    --decoder_layer_size ${decoder_layer_size} \
    --valid_every_epoch ${valid_every_epoch} \
    --attention ${attention} \
    --lstm ${lstm} \
    --decoder_embed_size ${decoder_embed_size} \
    --hidden_size ${hidden_size} \
    --label_smoothing ${ls} \
    --att_reg 0 \
    --dropout ${dropout} \
    --patience ${patience} \
    --optimizer ${optimizer} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --clip_grad_mode ${clip_grad_mode} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --beam_size ${beam_size} \
    --decode_max_time_step 100 \
    --log_every 50 \
    --save_to ${model_dir}${model_name} \
    #--load_model /data/saved_models/mtop/en/model.mtop.sup.lstm.hid768.de_embed256.drop0.3.lr_decay0.985.lr_dec_aft40.beam1.pat1000.max_ep240.batch64.lr0.001.glorot.ls0.seed.clip_grad_modenorm.parserpretrain_seq2seq_action.suffixscholarsubgeo.bin \
    #--sup_attention
    #--glove_embed_path embedding/glove/glove.6B.200d.txt \
    #--sup_attention \
    #--dev_file datasets/jobs/test.bin \
    #2>logs/geo/question_split/question_split/${model_name}.log
#--load_model /data/saved_models/mtop/en/model.mtop.sup.lstm.hid768.de_embed256.drop0.3.lr_decay0.985.lr_dec_aft40.beam1.pat1000.max_ep240.batch64.lr0.001.glorot.ls0.seed.clip_grad_modenorm.parserpretrain_seq2seq_action.suffixsup_att_ls.bin \

#./scripts/android_eval/test.sh ${model_dir}${model_name}.bin

#./scripts/multi_tasks/test.sh ${model_dir}${model_name}.bin multi_en_pt_test

#./scripts/multi_tasks/test.sh ${model_dir}${model_name}.bin multi_jobs_test

#./scripts/multi_tasks/test.sh ${model_dir}${model_name}.bin multi_question_test
cd server
python app.py --config config/config_files


#genienlp train \
#--data datadir --train_tasks_names almond --save model --no_commit --skip_cache --exist_ok \
#--train_iterations 80000 --log_every 100 --save_every 1000 --val_every 1000 --preserve_case \
#--dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=bert-base-uncased \
#--decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 \
#--rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.1 --train_batch_tokens 9000 \
#--append_question_to_context_too --val_batch_size 256
