from __future__ import print_function
import os, sys

import nltk
from transformers import AutoTokenizer

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import time

import six
import argparse
import subprocess

from flask import Flask, jsonify, render_template, request
import json


from evaluation import evaluate
from common.registerable import Registrable
from components.dataset import Example
from model.pretrain_seq2seq_ui import Seq2SeqModel
from components.evaluator import DefaultEvaluator

import gzip

app = Flask(__name__)
parsers = None
ALLOWED_EXTENSIONS = ['txt', 'tsv', 'gz']
UPLOAD_FOLDER = 'datasets/prolog'
train_file = 'UttrenceTrainnig.txt'

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--cuda', action='store_true', default=True, help='Use gpu')
    arg_parser.add_argument('--config_file', type=str, required=True,
                            help='Config file that specifies model to load, see online doc for an example')
    arg_parser.add_argument('--port', type=int, required=False, default=8081)

    return arg_parser


def decode(examples, model, decode_max_time_step, beam_size, verbose=False):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    decode_results = []
    for example in examples:
        hyps = model.beam_search(example, decode_max_time_step, beam_size=beam_size)
        decode_results.extend(hyps)
    if was_training: model.train()

    return decode_results


@app.route('/parse/<setting>/', methods=['GET'])
def parse(setting):
    input_str = request.args['q']
    #print(utterance)
    parser = parsers[setting]
    tokenizer = tokenizers[setting]

    if six.PY2:
        input_str = input_str.encode('utf-8', 'ignore')

    input_str_list = input_str.split('|||')
    utterance = input_str_list[0]
    button_list = [tokenizer.pad_token] + [schema_token.strip() for schema_token in input_str_list[1].split(',') if schema_token]

    print("==========================")
    print(utterance)
    print(button_list)

    decode_results = decode([Example(tokenizer.tokenize(utterance), [], [], [], idx=0, meta=None, schema_list=[tokenizer.tokenize(button_name) for button_name in button_list])], parser, 100, 5, verbose=False)

    responses = dict()
    responses['hypotheses'] = []

    cnt = 0
    for hyp_id, hyp in enumerate(decode_results):
        if config['parser_type'] == 'pretrain_seq2seq_ui':
            print('------------------ Hypothesis %d ------------------' % (cnt + 1))
            lf = hyp.to_ui_logic_form(tokenizer)
            print(lf)
            if '<pad>' in lf:
                continue
            hyp_entry = dict(id=cnt + 1,
                             value=lf,
                             score=hyp.score)

            responses['hypotheses'].append(hyp_entry)
        else:
            print('------------------ Hypothesis %d ------------------' % (cnt + 1))
            print(hyp.to_logic_form)

            hyp_entry = dict(id=cnt + 1,
                             value=hyp.to_logic_form,
                             score=hyp.score)

            responses['hypotheses'].append(hyp_entry)
        cnt += 1

    return jsonify(responses)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def update_parser():
    print('update_parser::start')
    global parsers
    global tokenizers
    global plmm_types
    parsers = dict()
    tokenizers = dict()
    plmm_types = dict()

    args = init_arg_parser().parse_args()
    config_dict = json.load(open(args.config_file))

    global config

    for setting, config in config_dict.items():
        parser_id = config['parser_type']
        parser_cls = Registrable.by_name(parser_id)
        parser, src_vocab, vertex_vocab = parser_cls.load(model_path=config['model_path'], use_cuda=args.cuda, args=args)
        parser.eval()

        parsers[setting] = parser

        if parser_id.startswith('pretrain_seq2seq'):
            print("====================================== use Roberta tokenizer ========================")
            tokenizers[setting] = AutoTokenizer.from_pretrained(config['plmm_name'])
            plmm_types[setting] = config['plmm_name']
        else:
            tokenizers[setting] = nltk.TweetTokenizer()


if __name__ == '__main__':
    update_parser()
    app.run(host='0.0.0.0', port=8099, debug=True)
