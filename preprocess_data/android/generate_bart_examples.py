import ast
import json
import random
import sys
from collections import Counter

from nltk import TweetTokenizer
from transformers import BertTokenizer, AutoConfig, AutoTokenizer

from components.dataset import Example
from grammar.consts import UI_BUTTON
from preprocess_data.android.generate_android_examples import generate_dir

sys.path.append('./')
from components.vocab import TokenVocabEntry, Vocab
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry
import os
import sys
from itertools import chain
import re

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np


nltk_tokenizer = TweetTokenizer()



def produce_data(data_filepath, plmm_tokenizer):
    #plmm_tokenizer.add_tokens(['_V0_','_V1_','_V2_','_V3_','_V4_','_V5_','_V6_','_V7_','_V9_','_V10_','_V11_','_V12_','_V13_'],special_tokens=True)
    example = []
    tgt_code_list = []
    template_code_list = []
    src_list = []
    schema_list = []
    with open(data_filepath) as json_file:
        for line in json_file:
            line_list = line.split('\t')
            #print(line_list)
            src = line_list[0]

            tgt = line_list[1]

            current_schema = ast.literal_eval(line_list[2].strip())

            tgt = tgt.strip()

            tgt_split = tgt.split(' ')

            new_tgt_split = []
            input_text_tokens = []
            flag = False
            for tgt_token in tgt_split:
                if tgt_token.startswith('\'') and (not flag):
                    new_tgt_split.append('\'')
                    flag = True
                elif tgt_token.startswith('\'') and flag:
                    new_tgt_split.extend(plmm_tokenizer.tokenize(" ".join(input_text_tokens)))
                    new_tgt_split.append('\'')
                    input_text_tokens = []
                    flag = False
                elif flag:
                    input_text_tokens.append(tgt_token)
                else:
                    new_tgt_split.append(tgt_token)

            tgt_split_tokens = []
            tgt_split_template = []


            for token in new_tgt_split:
                if " ".join(token.split('_')) in current_schema and "PRESS" in tgt:
                    tgt_split_tokens.append(UI_BUTTON)
                    tgt_split_template.append(token)
                else:
                    tgt_split_tokens.append(token)
                    tgt_split_template.append(plmm_tokenizer.pad_token)

            #if not schema_idx == -1:
            #    current_schema = current_schema + ui_button_list_list[schema_idx]

            schema_tokenized = []
            for schema_token in current_schema:
                str_bpes = plmm_tokenizer.tokenize(schema_token, max_length=32, truncation=True)
                token_ids = plmm_tokenizer.convert_tokens_to_ids(str_bpes)
                token_str = plmm_tokenizer.decode(token_ids)
                schema_tokenized.append(str_bpes)


            schema_list.append(schema_tokenized)
            #src_split = plmm_tokenizer.tokenize(src, max_length=32, truncation=True)
            tgt_code_list.append(tgt)
            #print(src_split)


            src_list.append(src)
            template_code_list.append(tgt_split_template)
            #tgt_code_list.append(tgt_split_tokens)
            assert len(tgt_split_template) == len(tgt_split_tokens)

    json_file.close()




    assert len(src_list) == len(tgt_code_list), "instance numbers should be consistent"

    for i, src_sent in enumerate(src_list):
        example.append(
            Example(src_sent=src_sent, tgt_code=tgt_code_list[i], tgt_ast=[],
                    tgt_actions=[],
                    idx=i, meta=None, att_tokens=[], tgt_template=[], schema_list=[])) #template_code_list[i], schema_list=schema_list[i]))
    return example


def read_ui_button_list(data_filepath):
    ui_button_list_list = []
    ui_button_dict = {}
    with open(data_filepath) as json_file:
        cnt = 0
        for line in json_file:
            ui_list = ast.literal_eval(line)
            ui_button_list_list.append(ui_list)

            for ui_button in ui_list:
                ui_button_str = "_".join(ui_button.split(' '))
                if ui_button_str in ui_button_dict:
                    ui_button_dict[ui_button_str].append(cnt)
                else:
                    ui_button_dict[ui_button_str] = []
                    ui_button_dict[ui_button_str].append(cnt)

            cnt += 1
    return ui_button_list_list, ui_button_dict

def prepare_mtop(train_file, test_file, ui_button_path, dump_path_prefix):


    plmm_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

    #ui_button_list_list, ui_button_dict = read_ui_button_list(ui_button_path)

    #vocab_freq_cutoff = 0
    train_set = produce_data(train_file, plmm_tokenizer)

    #dev_set = produce_data(dev_file, plmm_tokenizer)

    test_set = produce_data(test_file, plmm_tokenizer)

    #de_train_set = produce_data(de_train_file)
    #train_set = train_set + de_train_set
    #produce_data(test_file)


    src_vocab = TokenVocabEntry()
    src_vocab.token2id = plmm_tokenizer.vocab
    src_vocab.id2token = {v: k for k, v in src_vocab.token2id.items()}

    #print(src_vocab.token2id['[SEP]'])

    src_vocab.pad_id = plmm_tokenizer.pad_token_id

    src_vocab.unk_id = plmm_tokenizer.unk_token_id
    # generate vocabulary for the code tokens!

    code_tokens = [e.tgt_code for e in train_set]

    #code_tokens = [e.tgt_code for e in train_set]

    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=10000, freq_cutoff=0)

    vocab = Vocab(source=src_vocab, code=code_vocab)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)


    print('Train set len: %d' % len(train_set))
    print('Test set len: %d' % len(test_set))


    pickle.dump(train_set, open(os.path.join(dump_path_prefix, 'train.bin'), 'wb'))

    pickle.dump(test_set, open(os.path.join(dump_path_prefix, 'test.bin'), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix, 'vocab.freq2.bin'), 'wb'))



if __name__ == '__main__':
    #en_path_prefix = "../../datasets/mtop/en/"
    lang = "en"
    path_prefix = "../../datasets/android_eval/"
    dump_path_prefix = "../../datasets/android_eval/"
    #generate_dir(en_path_prefix)
    generate_dir(path_prefix)
    generate_dir(dump_path_prefix)

    ui_button_path = os.path.join("eval_supplementary_files", 'ui_button_list.txt')

    train_file = os.path.join(path_prefix, 'train.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    #de_train_file = os.path.join(de_path_prefix, 'train.txt')

    prepare_mtop(train_file, test_file, ui_button_path, dump_path_prefix)
    pass
