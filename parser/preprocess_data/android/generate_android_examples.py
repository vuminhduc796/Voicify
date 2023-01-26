import ast
import json
import random
import sys
from collections import Counter

from nltk import TweetTokenizer
from transformers import BertTokenizer, AutoConfig, AutoTokenizer
sys.path.append('./')
from components.dataset import Example
from grammar.consts import UI_BUTTON


from components.vocab import TokenVocabEntry, Vocab

import os
import sys

try:
    import cPickle as pickle
except:
    import pickle


nltk_tokenizer = TweetTokenizer()

def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])

def produce_data(data_filepath, plmm_tokenizer):
    action_set = set(["(", ")", ",", "PRESS", "OPEN", "SWIPE"])
    #plmm_tokenizer.add_tokens(['_V0_','_V1_','_V2_','_V3_','_V4_','_V5_','_V6_','_V7_','_V9_','_V10_','_V11_','_V12_','_V13_'],special_tokens=True)
    example = []
    tgt_code_list = []
    template_code_list = []

    src_list = []
    lower_src_list = []

    schema_list = []
    att_token_list = []
    with open(data_filepath) as json_file:
        for line in json_file:
            #line = "write what is the best route from mom's house to the madison airport with snow tomorrow at 4pm	( ENTER , ' what is the best route from mom's house to the madison airport with snow tomorrow at 4pm ' )	['<pad>']"
            line_list = line.split('\t')
            #print(line_list)
            src = line_list[0]

            lower_src = line_list[0].lower()

            tgt = line_list[1]

            current_schema = ast.literal_eval(line_list[2].strip())

            tgt = tgt.strip()

            tgt_split = tgt.split(' ')

            new_tgt_split = []
            input_text_tokens = []
            input_text = ""
            flag = False
            for tgt_token in tgt_split:
                if tgt_token == '\'' and (not flag):
                    new_tgt_split.append('\'')
                    flag = True
                elif tgt_token == '\'' and flag:
                    new_tgt_split.extend(plmm_tokenizer.tokenize(" ".join(input_text_tokens)))
                    input_text = " ".join(input_text_tokens)
                    new_tgt_split.append('\'')
                    input_text_tokens = []
                    flag = False
                elif flag:
                    input_text_tokens.append(tgt_token)
                else:
                    new_tgt_split.append(tgt_token)

            tgt_split_tokens = []
            tgt_split_template = []

            action_tokens = []
            for token in new_tgt_split:
                if ("PRESS" in tgt or "OPEN" in tgt or "SWIPE" in tgt) and (not (token in action_set)):
                    action_tokens.append(plmm_tokenizer.tokenize(" ".join(token.lower().split(':')[-1].split('_'))))
                else:
                    action_tokens.append([])


                if " ".join(token.split('_')) in current_schema and "PRESS" in tgt:
                    tgt_split_tokens.append(UI_BUTTON)
                    tgt_split_template.append(token)
                else:
                    tgt_split_tokens.append(token)
                    tgt_split_template.append(plmm_tokenizer.pad_token)

            att_token_list.append(action_tokens)
            #if not schema_idx == -1:
            #    current_schema = current_schema + ui_button_list_list[schema_idx]

            schema_tokenized = []
            for schema_token in current_schema:
                str_bpes = plmm_tokenizer.tokenize(schema_token, max_length=128, truncation=True)
                token_ids = plmm_tokenizer.convert_tokens_to_ids(str_bpes)
                token_str = plmm_tokenizer.decode(token_ids)
                schema_tokenized.append(str_bpes)


            schema_list.append(schema_tokenized)

            if input_text:
                freq = src.count(input_text)
                #if input_text == 'write':
                #    print("============")
                src_split_tuple = split(src, input_text, freq) #src.split(input_text, maxsplit=1)
                src_split = plmm_tokenizer.tokenize(src_split_tuple[0], max_length=128, truncation=True) + plmm_tokenizer.tokenize(input_text, max_length=128, truncation=True)
                if src_split_tuple[1]:
                    src_split = src_split + plmm_tokenizer.tokenize(src_split_tuple[1], max_length=128, truncation=True)
            else:
                src_split = plmm_tokenizer.tokenize(src, max_length=128, truncation=True)

            #print(src_split)


            src_list.append(src_split)
            template_code_list.append(tgt_split_template)
            tgt_code_list.append(tgt_split_tokens)
            assert len(tgt_split_template) == len(tgt_split_tokens)

    json_file.close()




    assert len(src_list) == len(tgt_code_list), "instance numbers should be consistent"

    for i, src_sent in enumerate(src_list):
        example.append(
            Example(src_sent=src_sent, tgt_code=tgt_code_list[i], tgt_ast=[],
                    tgt_actions=[],
                    idx=i, meta=None, att_tokens=att_token_list[i], tgt_template=template_code_list[i], schema_list=schema_list[i]))
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


def remove_quote(tgt_code_list):
    new_tgt_code_list = []
    for tgt_code in tgt_code_list:
        new_tgt_code = []
        flag = True
        for code_idx, tgt_token in enumerate(tgt_code):
            if flag:
                new_tgt_code.append(tgt_token)
            if tgt_token == '\'' and flag and tgt_code[code_idx - 1] == ',':
                flag = False
            elif tgt_token == '\'' and (not flag) and tgt_code[code_idx+ 1] == ')':
                flag = True
                new_tgt_code.append(tgt_token)
        new_tgt_code_list.append(new_tgt_code)
    return new_tgt_code_list

def prepare_examples(train_file, test_file, dump_path_prefix, plmm_name):


    plmm_tokenizer = AutoTokenizer.from_pretrained(plmm_name)


    train_set = produce_data(train_file, plmm_tokenizer)


    if os.path.isfile(test_file):
        test_set = produce_data(test_file, plmm_tokenizer)




    src_vocab = TokenVocabEntry()
    src_vocab.token2id = plmm_tokenizer.vocab
    src_vocab.id2token = {v: k for k, v in src_vocab.token2id.items()}


    src_vocab.pad_id = plmm_tokenizer.pad_token_id

    src_vocab.unk_id = plmm_tokenizer.unk_token_id
    # generate vocabulary for the code tokens!

    code_tokens = remove_quote([e.tgt_code for e in train_set])



    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=10000, freq_cutoff=0)

    ui_button_tokens = [e.tgt_template for e in train_set]



    ui_button_vocab = TokenVocabEntry.from_corpus(ui_button_tokens, size=10000, freq_cutoff=0)

    vocab = Vocab(source=src_vocab, code=code_vocab, ui_button=ui_button_vocab)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)


    print('Train set len: %d' % len(train_set))
    if os.path.isfile(test_file):
        print('Test set len: %d' % len(test_set))


    pickle.dump(train_set, open(os.path.join(dump_path_prefix, 'train.bin'), 'wb'))
    if os.path.isfile(test_file):
        pickle.dump(test_set, open(os.path.join(dump_path_prefix, 'test.bin'), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix, 'vocab.freq2.bin'), 'wb'))

def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    #en_path_prefix = "../../datasets/mtop/en/"
    lang = "en"
    path_prefix = "../../datasets/android_user/"
    dump_path_prefix = "../../datasets/android_user/"
    plmm_name = 'xlm-roberta-base'
    #generate_dir(en_path_prefix)
    generate_dir(path_prefix)
    generate_dir(dump_path_prefix)

    train_file = os.path.join(path_prefix, 'train.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    #de_train_file = os.path.join(de_path_prefix, 'train.txt')

    prepare_examples(train_file, test_file, dump_path_prefix, plmm_name)
    pass
