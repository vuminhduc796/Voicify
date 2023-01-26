# coding=utf-8
import math
import sys
from collections import namedtuple

import numpy as np
import pickle
import torch.nn.functional as F
import torch

from grammar.vertex import RuleVertex, CompositeTreeVertex
from model import nn_utils
from grammar.rule import extract_action_lit
from common.utils import cached_property
from grammar.action import *
import random
from math import gcd
import operator
from collections import Counter
from grammar.utils import is_var
from grammar.consts import VAR_NAME, IMPLICIT_HEAD, ROOT, UI_BUTTON
import re

class Dataset(object):
    def __init__(self, examples, curriculum = False):
        self.examples = examples

        self.domain = None
        self.known_domains = []
        self.known_test_data_length = []

        if curriculum:
            examples.sort(key=lambda e: -len([token for token in e.tgt_code if token.startswith('[')]))

        (self.class_examples, self.class_idx) = self.generate_class_examples()
        if len(examples)>0 and len(examples[0].tgt_actions) > 0:


            # for l, examples in self.class_examples.items():
                #print (l)
                #print (len(examples))

            (self.template_map, self.template_instances) = self._read_by_templates(self.examples)

        if len(examples) > 0 and hasattr(examples[0],'schema_list') and len(examples[0].schema_list) > 0:
            self.schema_map = self._read_by_schema(examples)


    def _read_by_schema(self, examples):
        schema_map = dict()
        for example in examples:
            schema_str = " ".join([str(schema) for schema in example.schema_list])
            #print(schema_str)
            if schema_str in schema_map:
                schema_map[schema_str].append(example)
            else:
                schema_map[schema_str] = []
                schema_map[schema_str].append(example)
        return schema_map


    def _read_by_templates(self, examples):
        template_map = dict()
        template_instances = []
        for example in examples:
            template = example.tgt_ast.to_lambda_expr if example.tgt_ast.to_lambda_expr else example.to_logic_form
            # example.tgt_ast_t_seq.to_lambda_expr
            template_instances.append(template)
            if template in template_map:
                template_map[template].append(example)
            else:
                template_map[template] = []
                template_map[template].append(example)
        return template_map, template_instances

    def generate_class_examples(self):
        class_examples = {}
        class_idx = {}
        # print (len(examples))
        for idx, e in enumerate(self.examples):
            # print (set(e.tgt_actions))
            for action in set(e.tgt_actions):
                if action in class_examples:
                    class_examples[action].append(e)
                    class_idx[action].append(idx)
                else:
                    class_examples[action] = []
                    class_examples[action].append(e)
                    class_idx[action] = []
                    class_idx[action].append(idx)

        for lab, lab_examples in class_examples.items():
            # print (len(lab_examples))
            # print (lab)
            lab_examples.sort(key=lambda e: -len(e.src_sent))
        return class_examples, class_idx

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @property
    def all_actions(self):
        return [e.tgt_actions for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    @staticmethod
    def from_bin_file_list(file_path_list):
        examples = []
        for file_path in file_path_list:
            examples.extend(pickle.load(open(file_path, 'rb')))
        return Dataset(examples)

    def add(self, new_dataset):
        self.examples.extend(new_dataset.examples)
        self.class_examples = self.generate_class_examples()
        self.template_map, self.template_instances = self._read_by_templates(self.examples)

    def add_examples(self, new_examples):
        self.examples.extend(new_examples)
        self.class_examples = self.generate_class_examples()
        self.template_map, self.template_instances = self._read_by_templates(self.examples)

    def batch_iter(self, batch_size, shuffle=False, sort=True):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            #print (batch_id)
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            if sort:
                batch_examples.sort(key=lambda e: -len(e.src_sent))
            #if len(batch_examples) == batch_size:
            yield batch_examples

    def schema_batch_iter(self, batch_size, sort=True,  shuffle=True):

        #print(len(self.schema_map.keys()))

        #for key in self.schema_map.keys():
        #    print(key)
        #    print(len(self.schema_map[key]))

        schema_list = list(self.schema_map.items())
        random.shuffle(schema_list)
        #print(schema_list[0][0])
        for schema, examples_per_schema in schema_list:
            #print(schema)
            #print(examples_per_schema)
            index_arr = np.arange(len(examples_per_schema))
            if shuffle:
                np.random.shuffle(index_arr)

            batch_num = int(np.ceil(len(examples_per_schema) / float(batch_size)))
            for batch_id in range(batch_num):
                # print (batch_id)
                batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
                batch_examples = [examples_per_schema[i] for i in batch_ids]
                if sort:
                    batch_examples.sort(key=lambda e: -len(e.src_sent))
                # if len(batch_examples) == batch_size:
                yield batch_examples


    def random_sample_batch_iter(self, sample_size):
        index_arr = np.arange(len(self.examples))
        # print (index_arr)
        np.random.shuffle(index_arr)

        batch_ids = index_arr[:sample_size]
        # print(batch_ids)
        batch_examples = [self.examples[i] for i in batch_ids]

        return batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class Example(object):
    def __init__(self, src_sent, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None, att_tokens=[], tgt_template=[], raw_src_sent=[], schema_list = []):
        # str list
        self.src_sent = src_sent
        # str list
        self.tgt_code = tgt_code
        # vertext root
        self.tgt_ast = tgt_ast
        self.schema_list = schema_list
        # action sequence
        self.tgt_actions = tgt_actions
        self.tgt_template = tgt_template
        self.raw_src_sent = raw_src_sent

        self.idx = idx
        self.meta = meta

        self.att_tokens = att_tokens


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return " ".join(self.src_sent)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def copy(self):
        new_src_sent = [token for token in self.src_sent]
        new_tgt_code = [token for token in self.tgt_code]
        new_tgt_ast = self.tgt_ast.copy()
        new_tgt_actions = [action.copy() for action in self.tgt_actions]
        new_idx = self.idx
        new_meta = self.meta

        return Example(new_src_sent, new_tgt_actions ,new_tgt_code,  new_tgt_ast, idx=new_idx, meta=new_meta)


    @property
    def to_prolog_template(self):
        return self.tgt_ast.to_prolog_expr

    @property
    def to_lambda_template(self):
        return self.tgt_ast.to_lambda_expr

    @property
    def to_ui_logical_form(self):

        new_tgt_code = []

        for code_idx, code in enumerate(self.tgt_code):
            if self.tgt_template[code_idx] == '<pad>':
                new_tgt_code.append(code)
            else:
                new_tgt_code.append(self.tgt_template[code_idx])
        return " ".join([str(code) for code in new_tgt_code])


    @property
    def to_logic_form(self):
        return " ".join([str(code) for code in self.tgt_code])

    @property
    def tgt_ast_seq(self):
        tgt_ast_seq = []
        for action in self.tgt_actions:
            if isinstance(action, GenAction):
                tgt_ast_seq.append(action.vertex)
            elif isinstance(action, ReduceAction):
                tgt_ast_seq.append(action.rule.head)
            else:
                raise ValueError

        return tgt_ast_seq

    @property
    def tgt_ast_t_seq(self):
        tgt_ast_t_seq = self.tgt_ast.copy()

        visited, queue = set(), [tgt_ast_t_seq]

        while queue:
            vertex = queue.pop(0)
            v_id = id(vertex)
            visited.add(v_id)
            idx = 0
            for child in vertex.children:
                if id(child) not in visited:
                    parent_node = vertex
                    if isinstance(child, RuleVertex):
                        child_vertex = child

                        if child_vertex.original_var is not None:
                            parent_node.children[idx] = child_vertex.original_var
                            child_vertex.original_var.parent = parent_node
                        else:
                            child_vertex.head = ROOT
                            child_vertex.is_auto_nt = True
                            child.is_auto_nt = True
                            queue.append(child_vertex)
                    elif isinstance(child, CompositeTreeVertex):
                        self.reduce_comp_node(child.vertex)

                idx += 1

        return tgt_ast_t_seq

    def reduce_comp_node(self, vertex):
        visited, queue = set(), [vertex]

        while queue:
            vertex = queue.pop(0)
            v_id = id(vertex)
            visited.add(v_id)
            idx = 0
            for child in vertex.children:
                if id(child) not in visited:
                    parent_node = vertex
                    child_vertex = child

                    if child_vertex.original_var is not None:
                        parent_node.children[idx] = child_vertex.original_var
                        child_vertex.original_var.parent = parent_node
                    else:
                        queue.append(child_vertex)
                idx += 1



class Batch(object):
    def __init__(self, examples, vocab, training=True, append_boundary_sym=True, use_cuda=False, data_type='overnight', copy=False, tgt_lang = 'default', schema_list = [], tokenizer=None):

        self.copy = copy

        self.data_type = data_type
        self.examples = examples

        # source token seq
        self.src_sents = [e.src_sent for e in self.examples]

        self.src_sents_len = [len(e.src_sent) + 2 if append_boundary_sym else len(e.src_sent) for e in self.examples]

        self.schema_list = schema_list #[ for schema in schema_list]

        self.schema_len = [len(schema_tokens) + 2 if append_boundary_sym else len(schema_tokens) for schema_tokens in schema_list]

        self.schema_code = [["_".join(tokenizer.decode(tokenizer.convert_tokens_to_ids(schema_tokens)).split(' '))] for schema_tokens in schema_list]

        #print(self.src_sents)
        #print(self.src_sents_len)

        # target token seq
        self.tgt_code = [e.tgt_code for e in self.examples]

        self.tgt_code_len = [len(e.tgt_code) + 2 if append_boundary_sym else len(e.tgt_code) for e in self.examples]

        self.max_tgt_code_len = max(self.tgt_code_len)

        if hasattr(self.examples[0], 'tgt_template'):
            self.tgt_template = [e.tgt_template for e in self.examples]


        self.action_seq = [e.tgt_actions for e in self.examples]
        self.action_seq_len = [len(e.tgt_actions) + 1 if append_boundary_sym else len(e.tgt_actions) for e in
                               self.examples]
        self.max_action_num = max(self.action_seq_len)
            # max(self.action_seq_len)

        # max(self.action_seq_len)

        # action seq
        self.max_ast_seq_num = max(len(e.tgt_ast_seq) for e in self.examples)
        self.ast_seq = [e.tgt_ast_seq for e in self.examples]
        self.ast_seq_len = [len(e.tgt_ast_seq) + 1 if append_boundary_sym else len(e.tgt_ast_seq) for e in
                               self.examples]

        self.entity_seq = []
        for seq in self.action_seq:
            self.entity_seq.append([])
            for action in seq:
                self.entity_seq[-1].extend(action.entities)

        self.variable_seq = []
        for seq in self.action_seq:
            self.variable_seq.append([])
            for action in seq:
                self.variable_seq[-1].extend(action.variables)

        self.vocab = vocab
        self.use_cuda = use_cuda
        self.training = training
        self.append_boundary_sym = append_boundary_sym

        self.tgt_lang = tgt_lang


        if data_type == 'bert' or data_type == 'roberta':
            self.init_token_pointer_mask()

        if len(schema_list) > 0:
            self.init_ui_index_tensors(tokenizer)

    def __len__(self):
        return len(self.examples)


    # source sentence
    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='token')

    # source sentence
    @cached_property
    def schema_var(self):
        return nn_utils.to_input_variable(self.schema_list, self.vocab.source,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode=self.data_type)

    @cached_property
    def schema_code_var(self):
        return nn_utils.to_input_variable(self.schema_code, self.vocab.ui_button,
                                          use_cuda=self.use_cuda, append_boundary_sym=False,
                                          mode='token')

    def src_sents_span(self, predicate_tokens):
        return nn_utils.to_src_sents_span(self.src_sents, predicate_tokens)


    # source sentence
    @cached_property
    def bert_src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode=self.data_type)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    use_cuda=self.use_cuda)


    # target sequence
    @cached_property
    def tgt_seq_var(self):
        return nn_utils.to_input_variable(self.tgt_code, self.vocab.code,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='token')


    # target sequence
    @cached_property
    def tgt_multi_seq_var(self):
        return nn_utils.to_input_variable(self.tgt_code, self.vocab.code_dict[self.tgt_lang],
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='token')



    def init_token_pointer_mask(self):

        if not self.tgt_lang == 'default':
            tgt_vocab = self.vocab.code_dict[self.tgt_lang]
        else:
            tgt_vocab = self.vocab.code

        self.tgt_token_gen_mask = []

        self.tgt_token_copy_idx_mask = np.zeros((self.max_tgt_code_len, len(self), max(self.src_sents_len)), dtype='float32')

        tgt_codes = [['<s>'] + seq + ['</s>'] for seq in self.tgt_code]
        #print(tgt_codes)
        #print(self.copy)
        for t in range(self.max_tgt_code_len):
            gen_token_mask_row = []
            for e_id, e_code in enumerate(tgt_codes):
                gen_token_mask = 0
                if t < len(e_code):
                    token = e_code[t]

                    src_sent = self.src_sents[e_id]

                    #print(token)

                    token_idx = tgt_vocab[token]

                    token_can_copy = False

                    if self.copy and token in src_sent:
                        #print(token)
                        token_pos_list = [idx for idx, _token in enumerate(src_sent) if _token == token]
                        #print(src_sent)
                        #print(token)
                        #print(token_pos_list)
                        self.tgt_token_copy_idx_mask[t, e_id, token_pos_list] = 1.
                        token_can_copy = True

                    #print(token_can_copy)

                    if token_can_copy is False or ( not token_idx == tgt_vocab.unk_id ):
                        # if the token is not copied, we can only generate this token from the vocabulary,
                        # even if it is a <unk>.
                        # otherwise, we can still generate it from the vocabulary
                        gen_token_mask = 1


                gen_token_mask_row.append(gen_token_mask)

            self.tgt_token_gen_mask.append(gen_token_mask_row)


        T = torch.cuda if self.use_cuda else torch

        self.tgt_token_gen_mask = T.FloatTensor(self.tgt_token_gen_mask).to(dtype=torch.bool)
        #print(self.tgt_token_gen_mask)
        self.tgt_token_copy_idx_mask = torch.from_numpy(self.tgt_token_copy_idx_mask).to(dtype=torch.bool)

        if self.use_cuda: self.tgt_token_copy_idx_mask = self.tgt_token_copy_idx_mask.cuda()


    def init_overnight_index_tensors(self):
        self.nt_action_idx_matrix = []
        self.nt_action_mask = []
        self.t_action_idx_matrix = []
        self.t_action_mask = []

        for t in range(self.max_action_num):
            nt_action_idx_row = []
            nt_action_mask_row = []
            t_action_idx_matrix_row = []
            t_action_mask_row = []

            for e_id, e in enumerate(self.examples):
                nt_action_idx = nt_action_mask = t_action_idx = t_action_mask = 0
                if t < len(e.tgt_actions):
                    action = e.tgt_actions[t]

                    if isinstance(action, GenNTAction):
                        nt_action_idx = self.vocab.nt_action.token2id[action]

                        # assert self.grammar.id2prod[app_rule_idx] == action.production
                        nt_action_mask = 1
                    elif isinstance(action, GenTAction):
                        #print (self.vocab.t_action.token2id)
                        #print (action)
                        t_action_idx = self.vocab.t_action.token2id[action]
                        #print (self.vocab.primitive.id2word[0])
                        t_action_mask = 1
                    else:
                        raise ValueError

                nt_action_idx_row.append(nt_action_idx)
                #print (app_rule_idx_row)
                nt_action_mask_row.append(nt_action_mask)

                t_action_idx_matrix_row.append(t_action_idx)
                t_action_mask_row.append(t_action_mask)

            #print ("================")
            #print (app_rule_idx_row)
            #print (token_row)
            self.nt_action_idx_matrix.append(nt_action_idx_row)
            self.nt_action_mask.append(nt_action_mask_row)

            self.t_action_idx_matrix.append(t_action_idx_matrix_row)
            self.t_action_mask.append(t_action_mask_row)



        T = torch.cuda if self.use_cuda else torch
        self.nt_action_idx_matrix = T.LongTensor(self.nt_action_idx_matrix)
        self.nt_action_mask = T.FloatTensor(self.nt_action_mask)
        self.t_action_idx_matrix = T.LongTensor(self.t_action_idx_matrix)
        self.t_action_mask = T.FloatTensor(self.t_action_mask)


    def init_ui_index_tensors(self, tokenizer):
        self.ui_button_idx_matrix = []
        self.ui_button_mask = []


        #token_ids = plmm_tokenizer.convert_tokens_to_ids(str_bpes)
        #token_str = plmm_tokenizer.decode(token_ids)

        ori_schema_list = ["_".join(tokenizer.decode(tokenizer.convert_tokens_to_ids(schema)).split(' ')) for schema in self.schema_list]

        #print(ori_schema_list)

        tgt_codes = [['<s>'] + seq + ['</s>'] for seq in self.tgt_code]
        tgt_templates = [['<s>'] + seq + ['</s>'] for seq in self.tgt_template]
        #print(ori_schema_list)
        for t in range(self.max_tgt_code_len):
            ui_button_idx_row = []
            ui_button_mask_row = []

            for e_id, tgt_code in enumerate(tgt_codes):
                ui_button_idx = ui_button_mask = 0
                if t < len(tgt_code):
                    code_token = tgt_code[t]
                    ui_button = tgt_templates[e_id][t]
                    if code_token == UI_BUTTON:
                        ui_button_idx = ori_schema_list.index(ui_button)

                        ui_button_mask = 1

                ui_button_idx_row.append(ui_button_idx)
                #print (app_rule_idx_row)
                ui_button_mask_row.append(ui_button_mask)


            self.ui_button_idx_matrix.append(ui_button_idx_row)
            self.ui_button_mask.append(ui_button_mask_row)



        T = torch.cuda if self.use_cuda else torch
        self.ui_button_idx_matrix = T.LongTensor(self.ui_button_idx_matrix)
        self.ui_button_mask = T.FloatTensor(self.ui_button_mask)
