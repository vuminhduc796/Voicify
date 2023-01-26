from __future__ import print_function

import sys, traceback
import numpy as np
import torch
import sacrebleu
from transformers import AutoTokenizer

from grammar.utils import is_var, is_predicate, is_lit
from common.registerable import Registrable
from grammar.consts import SLOT_PREFIX, VAR_NAME, ROOT, IMPLICIT_HEAD
from grammar.hypothesis import Hypothesis
from grammar.vertex import RuleVertex

from sklearn.metrics import classification_report
from common.utils import config_logger
import os


logger = config_logger("evaluation_result")



@Registrable.register('default_evaluator')
class DefaultEvaluator(object):
    def __init__(self, args=None, vocab=None):
        self.args = args
        self.default_metric = 'accuracy'
        self.correct_num = 'correct_num'
        self.correct_array = 'correct_array'
        self.plmm_tokenizer = AutoTokenizer.from_pretrained(args.plmm_model_name)


    def is_hyp_correct(self, example, hyp):
        assert isinstance(hyp, Hypothesis), "hyp should be Hypothesis"

        if self.args.parser == 'pretrain_seq2seq_ui':

            if hyp.to_logic_form == example.to_ui_logical_form:
                return True
            else:

                print("=======================================================")
                print("Source Sentence")
                print(self.plmm_tokenizer.decode(self.plmm_tokenizer.convert_tokens_to_ids(example.src_sent)))
                print("Token Candidate")
                print(hyp.to_logic_form)
                print("Token Reference")
                print(example.to_ui_logical_form)
                print("=======================================================")

                return False

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        correct_array = []
        oracle_array = []
        correct_pairs = []


        for example, hyp_list in zip(examples, decode_results):
            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):


                    is_correct = self.is_hyp_correct(example, hyp)
                    #if is_correct:
                    if hasattr(example, 'raw_src_sent'):
                        correct_pairs.append((example.raw_src_sent, hyp.to_logic_form))
                    hyp.is_correct = is_correct

                correct_array.append(hyp_list[0].is_correct)
                oracle_array.append(any(hyp.is_correct for hyp in hyp_list))


            else:

                correct_array.append(False)
                oracle_array.append(False)


        torch_correct_array = torch.Tensor(correct_array)
        acc = np.average(correct_array)

        oracle_acc = np.average(oracle_array)
        eval_results = dict(accuracy=acc,
                            oracle_accuracy=oracle_acc, correct_num = np.sum(correct_array), correct_array = torch_correct_array, correct_pairs=correct_pairs)
        # print("Count : ", count)
        return eval_results

