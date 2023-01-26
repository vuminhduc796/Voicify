# coding=utf-8
from grammar.rule import ReduceAction
from grammar.vertex import CompositeTreeVertex, RuleVertex, TreeVertex
from grammar.rule import GenNTAction, GenTAction
from grammar.rule import ProductionRuleBLB
from model.attention_util import dot_prod_attention
import torch
from grammar.consts import IMPLICIT_HEAD, ROOT, FULL_STACK_LENGTH, NT, TYPE_SIGN, NT_TYPE_SIGN, STRING_FIELD, END_TOKEN


class Hypothesis(object):
    def __init__(self):
        self.tree = None
        # action
        self.actions = []
        self.action_id = []
        self.general_action_id = []
        # tgt code
        self.tgt_code_tokens = []
        self.tgt_code_tokens_id = []

        self.var_id = []
        self.ent_id = []

        self.heads_stack = []
        self.tgt_ids_stack = []
        self.heads_embedding_stack = []
        self.embedding_stack = []
        self.hidden_embedding_stack = []
        self.v_hidden_embedding = None

        self.current_gen_emb = None
        self.current_re_emb = None
        self.current_att = None
        self.current_att_cov = None

        self.score = 0.
        self.is_correct = False
        self.is_parsable = True
        # record the current time step
        self.reduce_action_count = 0
        self.t = 0

        self.frontier_node = None
        self.frontier_field = None


    @property
    def to_prolog_template(self):
        if self.tree:
            return self.tree.to_prolog_expr
        else:
            return ""

    @property
    def to_lambda_template(self):
        if self.tree:
            return self.tree.to_lambda_expr
        else:
            return ""

    def to_ui_logic_form(self, tokenizer):

        new_tgt_code = []
        input_text_tokens = []
        flag = True
        for code_idx, tgt_token in enumerate(self.tgt_code_tokens):
            if flag:
                new_tgt_code.append(tgt_token)
            else:
                input_text_tokens.append(tgt_token)
            if tgt_token == '\'' and flag and code_idx - 1 >=0 and self.tgt_code_tokens[code_idx - 1] == ',':
                flag = False
            elif tgt_token == '\'' and (not flag) and code_idx+ 1 < len(self.tgt_code_tokens) and self.tgt_code_tokens[code_idx+ 1] == ')':
                flag = True
                if input_text_tokens:
                    input_text_tokens = input_text_tokens[:-1]
                    input_token_ids = tokenizer.convert_tokens_to_ids(input_text_tokens)
                    input_token_str = tokenizer.decode(input_token_ids)
                    new_tgt_code.append(input_token_str)
                new_tgt_code.append(tgt_token)

        return " ".join([str(code) for code in new_tgt_code])

    @property
    def to_logic_form(self):
        return " ".join([str(code) for code in self.tgt_code_tokens])


    def copy(self):
        new_hyp = Hypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()
        new_hyp.action_id = list(self.action_id)
        new_hyp.actions = list(self.actions)
        new_hyp.general_action_id = list(self.general_action_id)
        new_hyp.var_id = list(self.var_id)
        new_hyp.ent_id = list(self.ent_id)
        new_hyp.heads_stack = list(self.heads_stack)
        new_hyp.tgt_ids_stack = list(self.tgt_ids_stack)
        new_hyp.embedding_stack = [embedding.clone() for embedding in self.embedding_stack]
        new_hyp.heads_embedding_stack = [embedding.clone() for embedding in self.heads_embedding_stack]
        new_hyp.hidden_embedding_stack = [(state.clone(), cell.clone()) for state, cell in self.hidden_embedding_stack]
        new_hyp.tgt_code_tokens_id = [token_id for token_id in self.tgt_code_tokens_id]
        new_hyp.tgt_code_tokens = [token for token in self.tgt_code_tokens]

        new_hyp.score = self.score
        new_hyp.t = self.t
        new_hyp.is_correct = self.is_correct
        new_hyp.is_parsable = self.is_parsable
        new_hyp.reduce_action_count = self.reduce_action_count

        if self.current_gen_emb is not None:
            new_hyp.current_gen_emb = self.current_gen_emb.clone()

        if self.current_re_emb is not None:
            new_hyp.current_re_emb = self.current_re_emb.clone()

        if self.current_att is not None:
            new_hyp.current_att = self.current_att.clone()

        if self.current_att_cov is not None:
            new_hyp.current_att_cov = self.current_att_cov.clone()

        if self.v_hidden_embedding is not None:
            new_hyp.v_hidden_embedding = (self.v_hidden_embedding[0].clone(), self.v_hidden_embedding[1].clone())

        return new_hyp

