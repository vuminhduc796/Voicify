# coding=utf-8
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn.functional as F

from components.dataset import Batch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from common.registerable import Registrable
from grammar.consts import UI_BUTTON
from model import nn_utils
from grammar.rule import ReduceAction
from grammar.hypothesis import Hypothesis
from common.utils import update_args, init_arg_parser
import os

from model.attention_util import AttentionUtil
from model.common import MultiLSTMCell
from model.identity_encoder import IdentityEncoder
from model.nn_utils import masked_log_softmax, MultiHeadAttention, masked_softmax
from model.pointer_net import PointerNet



@Registrable.register('pretrain_seq2seq_ui')
class Seq2SeqModel(nn.Module):
    """
    a standard seq2seq model
    """
    def __init__(self, vocab, args):
        super(Seq2SeqModel, self).__init__()
        self.use_cuda = args.use_cuda
        self.decoder_embed_size = args.decoder_embed_size
        self.hidden_size = args.hidden_size
        self.decoder_layer_size = args.decoder_layer_size
        self.vocab = vocab
        self.args = args

        self.sup_attention = args.sup_attention

        self.plmm_model_name = args.plmm_model_name

        self.plmm_config = AutoConfig.from_pretrained(self.plmm_model_name)  # Initialize tokenizer

        self.copy = args.copy
        self.src_vocab = vocab.source
        self.tgt_vocab = vocab.code

        self.tgt_emb = nn.Embedding(len(self.tgt_vocab), self.decoder_embed_size)
        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(self.plmm_config.hidden_size , self.hidden_size)

        self.decoder_lstm = MultiLSTMCell(self.decoder_layer_size, self.decoder_embed_size * 2, self.hidden_size, args.dropout)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        #self.att_src_linear = nn.Linear(self.plmm_config.hidden_size, self.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(self.hidden_size + self.hidden_size, self.decoder_embed_size, bias=False)


        self.readout_b = nn.Parameter(torch.FloatTensor(len(self.tgt_vocab)).zero_())

        self.readout = lambda q: F.linear(q, self.tgt_emb.weight, self.readout_b)

        self.readout_schema = nn.Linear(self.decoder_embed_size, self.hidden_size, bias=False)

        self.ui_button = vocab.ui_button
        self.schema_code_emb = nn.Embedding(len(self.ui_button), self.hidden_size)
        # prediction layer of the target vocabulary
        #self.readout = nn.Linear(self.hidden_size, len(self.tgt_vocab), bias=False)

        # dropout layer
        self.dropout_rate = args.dropout

        self.dropout = nn.Dropout(self.dropout_rate)
        self.decoder_word_dropout = args.decoder_word_dropout
        self.label_smoothing = None

        # label smoothing
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing:
            self.label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.tgt_vocab), ignore_indices=[0])

        self.multi_att_size = args.multi_att_size
        self.embed_type = args.embed_type  # Initialize tokenizer

        self.plmm_model = AutoModel.from_pretrained(self.plmm_model_name)
        self.plmm_model.resize_token_embeddings(len(self.src_vocab))
        self.plmm_tokenizer = AutoTokenizer.from_pretrained(self.plmm_model_name)



        self.encoder = IdentityEncoder(self.src_vocab, args, self.plmm_config, self.plmm_model)

        # pointer net to the source
        self.src_pointer_net = PointerNet(src_encoding_size=self.hidden_size,
                                          query_vec_size=self.decoder_embed_size)

        self.tgt_token_predictor = nn.Linear(self.decoder_embed_size, 2)

        self.attention = self.args.attention

        if self.attention == 'mhd':
            self.attention_layer = MultiHeadAttention(n_head = self.plmm_config.num_attention_heads, d_model = self.hidden_size, d_k = self.hidden_size//self.plmm_config.num_attention_heads, d_v = self.hidden_size//self.plmm_config.num_attention_heads)

        if args.use_cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.mask_dict = self.ui_mask()


    def ui_mask(self):
        mask_dict = {}


        default_tokens = ['(', ')', ',']

        mask_dict['SWIPE'] = []
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        for direction in directions:
            mask_dict['SWIPE'].append(self.tgt_vocab.token2id[direction])

        mask_dict['OPEN'] = []
        for token , tokenid in self.tgt_vocab.token2id.items():
            if token.startswith('app') or token.startswith('component'):
                mask_dict['OPEN'].append(tokenid)

        mask_dict['PRESS'] = []
        mask_dict['PRESS'].append(self.tgt_vocab.token2id[UI_BUTTON])

        mask_dict['NO_UI_BUTTON'] = []
        for token , tokenid in self.tgt_vocab.token2id.items():
            if not 'PRESS' in token:
                mask_dict['NO_UI_BUTTON'].append(tokenid)

        mask_dict['ACTIONS'] = []
        # 'OVERVIEW_BUTTON', 'BACK', 'HOME',
        actions = ['PRESS', 'LONG_PRESS', 'SWIPE', 'OPEN', 'DOUBLE_PRESS', 'ENTER']

        for act in actions:
            mask_dict['ACTIONS'].append(self.tgt_vocab.token2id[act])

        for action, mask in mask_dict.items():
            mask.extend([0,1,2,3])
            for default_token in default_tokens:
                mask.append(self.tgt_vocab.token2id[default_token])
        return mask_dict


    def encode(self, plmm_encodeing):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
            dec_init_state, dec_init_cell: Variable(batch_size, hidden_size)
        """

        # (tgt_query_len, batch_size, embed_size)
        # print (src_sents_var.size())
        src_encodings = plmm_encodeing

        last_state = src_encodings.mean(0)
        last_cell = src_encodings.mean(0)
        return src_encodings, (last_state, last_cell)

    def init_decoder_state(self, enc_last_state, enc_last_cell):
        dec_init_cell = self.decoder_cell_init(enc_last_cell)
        #print (dec_init_cell.squeeze().size())
        dec_init_state = torch.tanh(dec_init_cell)

        #print(dec_init_cell.size())
        return dec_init_state, dec_init_cell

    def decode(self, batch, src_encodings, src_sent_masks, dec_init_vec, tgt_sents_var):
        """
        compute the final softmax layer at each decoding step
        :param src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
        :param src_sents_len: list[int]
        :param dec_init_vec: tuple((batch_size, hidden_size))
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            scores: Variable(src_sent_len, batch_size, src_vocab_size)
        """
        new_tensor = src_encodings.data.new
        batch_size = src_encodings.size(0)

        #print(batch_size)

        h_tm1 = dec_init_vec
        # (batch_size, query_len, hidden_size * 2)
        #src_encodings = src_encodings.permute(1, 0, 2)

        #print(src_encodings.size())
        #print(src_encodings.size())
        # (batch_size, query_len, hidden_size)
        #if not self.plmm_config.hidden_size == self.hidden_size:
            #print(src_encodings.size())
        #    src_encodings_att_linear = self.att_src_linear(src_encodings)
        #else:
        src_encodings_att_linear = src_encodings

        #print(src_encodings_att_linear.size())
        # initialize the attentional vector
        att_tm1 = new_tensor(batch_size, self.decoder_embed_size).zero_()
        assert (att_tm1.requires_grad == False, "the att_tm1 requires grad is False")
        # (batch_size, src_sent_len)
        #print (src_sent_masks)
        #print (src_sent_masks.size())
        # (tgt_sent_len, batch_size, embed_size)
        tgt_token_embed = self.tgt_emb(tgt_sents_var)
        #print (tgt_token_embed.size())
        att_vecs = []

        att_probs = []

        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(tgt_token_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim
            y_tm1_embed = y_tm1_embed.squeeze(0)
            """
            if t > 0 and self.decoder_word_dropout:
                # (batch_size)
                y_tm1_mask = torch.bernoulli(new_tensor(batch_size).fill_(1 - self.decoder_word_dropout))
                y_tm1_embed = y_tm1_embed * y_tm1_mask.unsqueeze(1)
            """
            x = torch.cat([y_tm1_embed, att_tm1], 1)

            #print (src_sent_masks.size())
            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1,
                                                      src_encodings,
                                                      src_encodings_att_linear,
                                                      src_sent_masks=src_sent_masks)


            #print(att_weight.size())
            # if use supervised attention
            if self.sup_attention:
                for e_id, example in enumerate(batch.examples):
                    if t < len(example.tgt_code):
                        tgt_token_list = example.att_tokens[t]
                        cand_src_tokens = AttentionUtil.get_candidate_tokens_with_tgt_tokens(example.src_sent, tgt_token_list)

                        #print(tgt_token)
                        #print(cand_src_tokens)
                        if cand_src_tokens:
                            #print("==========================")
                            #print(tgt_token_list)
                            #print(cand_src_tokens)
                            #print(example.src_sent)
                            # print (att_weight[e_id].size())
                            # print (example.src_sent)
                            # print (att_weight[e_id])
                            # print (cand_src_tokens)
                            if self.attention == 'dot':
                                att_prob = [att_weight[e_id, token_id + 1].unsqueeze(0) for token_id in
                                            cand_src_tokens]
                            elif self.attention == 'mhd':
                                att_prob = [att_weight[e_id, :, token_id + 1].unsqueeze(0) for token_id in
                                            cand_src_tokens]
                            # print (cand_src_tokens)
                            # print (att_prob)
                            if len(att_prob) > 1:
                                #print(torch.cat(att_prob).size())
                                att_prob = torch.cat(att_prob).sum().unsqueeze(0)
                            else:
                                #print(att_prob[0].size())
                                att_prob = att_prob[0].sum().unsqueeze(0)
                            att_probs.append(att_prob)

            att_vecs.append(att_t)

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        # (src_sent_len, batch_size, tgt_vocab_size)
        att_vecs = torch.stack(att_vecs)


        if self.sup_attention:
            return att_vecs, att_probs
        else:
            return att_vecs


    def score_decoding_results(self, att_vecs, src_encodings, src_sent_masks, tgt_sents_var, batch, tgt_token_gen_mask, tgt_token_copy_idx_mask, schema_encodings):
        """
        :param scores: Variable(src_sent_len, batch_size, tgt_vocab_size)
        :param tgt_sents_var: Variable(src_sent_len, batch_size)
        :return:
            tgt_sent_log_scores: Variable(batch_size)
        """
        # schema_encodings: batch_size * ui_vocab_size * dimension
        # (tgt_sent_len, batch_size, tgt_vocab_size)

        #print(att_vecs.size())
        #print(batch.schema_code_var)
        schema_code_encoding = self.schema_code_emb(batch.schema_code_var).squeeze(0)
        #print("==========================")
        #print(schema_encodings.T.size())
        #print(schema_code_encoding.size())

        token_gen_prob = F.log_softmax(self.readout(att_vecs), dim=-1)

        #print(token_gen_prob.shape)
        #print(att_vecs.shape)
        #print(schema_encodings.T.shape)
        #breakpoint()
        schema_encoding_sum = schema_encodings.T + schema_code_encoding.T
        ui_button_prob = F.log_softmax(torch.matmul(self.readout_schema(att_vecs), schema_encoding_sum), dim=-1)

        #print(ui_button_prob.shape)

        if self.copy:
            tgt_token_predictor = F.log_softmax(self.tgt_token_predictor(att_vecs), dim=-1)
            log_token_copy_prob, _  = self.src_pointer_net(src_encodings, src_sent_masks, att_vecs)

        tgt_token_idx = tgt_sents_var[1:]  # remove leading <s>
        tgt_token_gen_mask = tgt_token_gen_mask[1:]
        tgt_token_copy_idx_mask = tgt_token_copy_idx_mask[1:]

        ui_button_idx = batch.ui_button_idx_matrix[1:]
        ui_button_mask = batch.ui_button_mask[1:]
        #print(tgt_token_idx)
        #print(tgt_token_idx.shape)
        #print(ui_button_idx)
        #print(ui_button_idx.shape)
        #print(ui_button_mask)
        #print(ui_button_mask.shape)
        #print(tgt_token_copy_idx_mask)
        #print(tgt_token_copy_idx_mask.shape)
        #print(tgt_token_gen_mask)
        #print(tgt_token_gen_mask.shape)


        if self.training and self.label_smoothing:
            # (tgt_sent_len, batch_size)
            tgt_token_gen_prob = -self.label_smoothing_layer(token_gen_prob, tgt_token_idx)

            ui_button_gen_prob = -self.label_smoothing_layer(ui_button_prob, ui_button_idx)

        else:
            # (tgt_sent_len, batch_size)
            tgt_token_gen_prob = torch.gather(token_gen_prob, dim=-1,
                                              index=tgt_token_idx.unsqueeze(-1)).squeeze(-1)

            ui_button_gen_prob = torch.gather(ui_button_prob, dim=-1,
                                              index=ui_button_idx.unsqueeze(-1)).squeeze(-1)


        if self.copy:
            tgt_token_gen_prob = tgt_token_gen_prob * tgt_token_gen_mask + (tgt_token_gen_mask + 1e-45).log()
            tgt_token_copy_prob = torch.logsumexp(log_token_copy_prob * tgt_token_copy_idx_mask + (tgt_token_copy_idx_mask + 1e-45).log(), dim=-1)


            tgt_token_mask = torch.gt(tgt_token_gen_mask + tgt_token_copy_idx_mask.sum(dim=-1), 0.).float()


            tgt_token_prob = torch.logsumexp(torch.stack([tgt_token_predictor[:, :, 0] + tgt_token_gen_prob,
                                       tgt_token_predictor[:, :, 1] + tgt_token_copy_prob], dim=0), dim=0)

            #print(ui_button_gen_prob * ui_button_mask)

            #print(tgt_token_prob * tgt_token_mask)

            tgt_token_prob = tgt_token_prob * tgt_token_mask + ui_button_gen_prob * ui_button_mask
        else:
            tgt_token_prob = tgt_token_gen_prob * tgt_token_gen_mask + ui_button_gen_prob * ui_button_mask

        # (batch_size)
        tgt_sent_log_scores = tgt_token_prob.sum(dim=0)


        return tgt_sent_log_scores



    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_sent_masks=None):
        """
        a single LSTM decoding step
        """
        # h_t: (batch_size, hidden_size)
        #print (h_tm1[1].size())
        #print(x.size())
        #print(h_tm1[0].size())
        dec_state, (h_t, cell_t) = self.decoder_lstm(x, h_tm1)
        #print(h_t.size())
        #print(h_t.size())
        #print(src_encodings.size())
        #print(src_sent_masks)
        if self.attention == 'mhd':
            mhd_mask = ~src_sent_masks if src_sent_masks is not None else None
            ctx_t, att_weight = self.attention_layer(q = dec_state, k = src_encodings, v = src_encodings_att_linear, mask=mhd_mask)
        else:
            ctx_t, att_weight = Seq2SeqModel.dot_prod_attention(dec_state, src_encodings, src_encodings_att_linear, mask=src_sent_masks)
        #print(h_t.size())
        #print(ctx_t.size())
        att_t = torch.tanh(self.att_vec_linear(torch.cat([dec_state, ctx_t], 1)))  # E.q. (5)

        att_t = self.dropout(att_t)


        return (h_t, cell_t), att_t, att_weight

    def forward(self, examples):
        """
        encode source sequence and compute the decoding log likelihood
        :param src_sents_var: Variable(src_sent_len, batch_size)
        :param src_sents_len: list[int]
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            tgt_token_scores: Variable(tgt_sent_len, batch_size, tgt_vocab_size)
        """
        schema_list = examples[0].schema_list
        #print(len(examples))
        #print(examples)
        #print(schema_list)
        #self.plmm_tokenizer
        batch = Batch(examples, self.vocab, use_cuda=self.use_cuda, data_type='roberta', copy=self.copy, schema_list=schema_list, tokenizer=self.plmm_tokenizer)


        sent_lengths = batch.src_sents_len


        sent_lengths_tensor = self.new_tensor(sent_lengths)


        src_encodings, (last_state, last_cell) = self.encoder(batch.bert_src_sents_var.T, sent_lengths_tensor)

        #if len(schema_list) > 0:
        schema_lengths = batch.schema_len


        schema_lengths_tensor = self.new_tensor(schema_lengths)


        schema_encodings, (schema_last_state, schema_last_cell) = self.encoder(batch.schema_var.T, schema_lengths_tensor)
        #print(schema_encodings.shape)
        schema_encodings_head = schema_encodings[:,0,:]
        tgt_seq_var = batch.tgt_seq_var

        #print(tgt_seq_var)
        #print(tgt_seq_var.size())
        #print(tgt_seq_var.size())

        #src_encodings, (last_state, last_cell) = self.encode(plmm_src_token_embed)
        #dec_init_vec = self.init_decoder_state(last_state, last_cell)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(sent_lengths, use_cuda=self.use_cuda)

        if self.sup_attention:
            tgt_token_logits, att_prob = self.decode(batch, src_encodings, src_sent_masks, (last_state, last_cell), tgt_seq_var)
        else:
            tgt_token_logits = self.decode(batch, src_encodings, src_sent_masks, (last_state, last_cell), tgt_seq_var)

        #print(tgt_seq_var.size())
        #print(batch.tgt_token_gen_mask.size())

        tgt_sent_log_scores = self.score_decoding_results(tgt_token_logits, src_encodings, src_sent_masks, tgt_seq_var, batch, tgt_token_gen_mask = batch.tgt_token_gen_mask, tgt_token_copy_idx_mask = batch.tgt_token_copy_idx_mask, schema_encodings = schema_encodings_head)
        #print(tgt_sent_log_scores.size())
        #print(tgt_sent_log_scores[0].size())
        loss = -tgt_sent_log_scores

        returns = [loss]
        #print(att_prob)
        #for prob in att_prob:
        #    print(prob.size())
        if self.sup_attention:
            #print(att_prob)
            returns.append(att_prob)

        return returns

    @staticmethod
    def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        att_weight_score = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)

        if mask is not None:
            #
            log_weight = masked_log_softmax(att_weight_score, ~mask, dim=-1)
            att_weight_score.data.masked_fill_(mask, -float('inf'))
        else:
            log_weight = torch.log_softmax(att_weight_score, dim=-1)
        #print(att_weight.size())
        att_weight = torch.softmax(att_weight_score, dim=-1)
        #print(att_weight)
        #print (att_weight)
        #print (torch.sum(att_weight[0]))
        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)


        return ctx_vec, log_weight

    def initilize_hyp(self, bos_id):
        first_hyp = Hypothesis()
        first_hyp.tgt_code_tokens_id.append(bos_id)
        first_hyp.tgt_code_tokens.append('<s>')
        return first_hyp


    def beam_search(self, example, decode_max_time_step, beam_size=5):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """

        #batch_src_sents = [" ".join(example.src_sent)]
        # print(batch_src_sents)
        #inputs = self.plmm_tokenizer(batch_src_sents, return_tensors='pt', padding=True, truncation=True,
        #                             return_length=True).to('cuda')

        #plmm_src_token_embed = self.plmm_model(inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True).last_hidden_state
        #plmm_src_token_embed = plmm_src_token_embed.permute(1, 0, 2)

        #TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
        #src_encodings, (last_state, last_cell) = self.encode(plmm_src_token_embed)
        # (1, query_len, hidden_size * 2)
        #src_encodings = src_encodings.permute(1, 0, 2)
        #h_tm1 = self.init_decoder_state(last_state, last_cell)

        # analyze which tokens can be copied from the source
        src_sent = example.src_sent
        src_token_tgt_vocab_ids = [self.tgt_vocab[token] for token in src_sent]
        src_unk_pos_list = [pos for pos, token_id in enumerate(src_token_tgt_vocab_ids) if
                            token_id == self.tgt_vocab.unk_id]
        schema_list = example.schema_list
        #print(schema_list)
        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        #token_set = set()
        #for i, tid in enumerate(src_token_tgt_vocab_ids):
        #    if tid in token_set:
        #        src_token_tgt_vocab_ids[i] = -1
        #    else:
        #        token_set.add(tid)


        new_float_tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        new_long_tensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

        #print(self.copy)

        batch = Batch([example], self.vocab, use_cuda=self.use_cuda, data_type='roberta', copy=self.copy, schema_list=schema_list, tokenizer=self.plmm_tokenizer)

        sent_lengths = batch.src_sents_len

        sent_lengths_tensor = new_float_tensor(sent_lengths)

        src_encodings, (last_state, last_cell) = self.encoder(batch.bert_src_sents_var.T, sent_lengths_tensor)


        schema_lengths = batch.schema_len

        schema_lengths_tensor = new_float_tensor(schema_lengths)

        schema_encodings, (schema_last_state, schema_last_cell) = self.encoder(batch.schema_var.T,
                                                                               schema_lengths_tensor)


        schema_encodings_head = schema_encodings[:,0,:]

        schema_code_encoding = self.schema_code_emb(batch.schema_code_var).squeeze(0)

        h_tm1 = (last_state, last_cell)
        # tensor constructors

        #if not self.plmm_config.hidden_size == self.hidden_size:
        #    print(src_encodings.size())
        #    src_encodings_att_linear = self.att_src_linear(src_encodings)
        #else:
        src_encodings_att_linear = src_encodings

        att_tm1 = torch.zeros(1, self.decoder_embed_size, requires_grad=False)
        hyp_scores = torch.zeros(1, requires_grad=False)
        if self.use_cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()
        # todo change it back
        eos_id = self.tgt_vocab['</s>']
        bos_id = self.tgt_vocab['<s>']
        tgt_vocab_size = len(self.tgt_vocab)




        first_hyp = Hypothesis()
        first_hyp.tgt_code_tokens_id.append(bos_id)
        first_hyp.tgt_code_tokens.append('<s>')
        #hypotheses = [[bos_id]]
        hypotheses = [first_hyp]
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))

            expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, src_encodings_att_linear.size(1), src_encodings_att_linear.size(2))

            #expanded_schema_encodings = schema_encodings.T.expand(hyp_num, src_encodings_att_linear.size(1), src_encodings_att_linear.size(2))

            y_tm1 = new_long_tensor([hyp.tgt_code_tokens_id[-1] for hyp in hypotheses])
            y_tm1_embed = self.tgt_emb(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, expanded_src_encodings,
                                                      expanded_src_encodings_att_linear,
                                                      src_sent_masks=None)




            # (batch_size, tgt_vocab_size)
            token_gen_prob = F.softmax(self.readout(att_t), dim=-1)
            #print(token_gen_prob.size())
            token_gen_prob[:, 0] = 0.0
            #print(self.readout_schema(att_t).shape)

            #print(schema_encodings_head.T.shape)

            ui_button_log_prob = F.log_softmax(torch.matmul(self.readout_schema(att_t), (schema_encodings_head.T + schema_code_encoding.T)), dim=-1)
            #print(ui_button_log_prob.shape)
            ui_button_log_prob[:, 0] = -1e20
            #print(ui_button_log_prob)
            if self.copy:
                tgt_token_predictor = F.softmax(self.tgt_token_predictor(att_t), dim=-1)
            # (batch_size, src_sent_len)
                _, token_copy_prob = self.src_pointer_net(src_encodings, src_token_mask=None, query_vec=att_t.unsqueeze(0))
                token_copy_prob = token_copy_prob.squeeze(0)
            #print(token_copy_prob.size())
            #print(token_copy_prob)

            # (batch_size, tgt_vocab_size)
                token_gen_prob = tgt_token_predictor[:, 0].unsqueeze(1) * token_gen_prob

                for token_pos, token_vocab_id in enumerate(src_token_tgt_vocab_ids):
                    if not token_vocab_id == self.tgt_vocab.unk_id:
                        p_copy = tgt_token_predictor[:, 1] * token_copy_prob[:, token_pos]
                        token_gen_prob[:, token_vocab_id] = token_gen_prob[:, token_vocab_id] + p_copy

                # second, add the probability of copying the most probable unk word
                gentoken_new_hyp_unks = []
                if src_unk_pos_list:
                    for hyp_id in range(hyp_num):
                        unk_pos = torch.argmax(token_copy_prob[hyp_id][src_unk_pos_list]).item()
                        unk_pos = src_unk_pos_list[unk_pos]
                        token = src_sent[unk_pos]
                        gentoken_new_hyp_unks.append(token)

                        unk_copy_score = tgt_token_predictor[hyp_id, 1] * token_copy_prob[hyp_id, unk_pos]
                        token_gen_prob[hyp_id, self.tgt_vocab.unk_id] = unk_copy_score




            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(token_gen_prob) + token_gen_prob.log()).view(-1)
            #print(new_hyp_scores.size())
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_new_hyp_pos // tgt_vocab_size

            word_ids = top_new_hyp_pos % tgt_vocab_size

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data.tolist(), word_ids.cpu().data.tolist(), top_new_hyp_scores.cpu().data.tolist()):
                #print(word_id)
                temp_hyp = hypotheses[prev_hyp_id].copy()
                temp_hyp.tgt_code_tokens_id.append(word_id)

                if word_id == self.tgt_vocab.unk_id:
                    if self.copy:
                        if gentoken_new_hyp_unks:
                            word = gentoken_new_hyp_unks[prev_hyp_id]
                        else:
                            word = self.tgt_vocab.id2token[self.tgt_vocab.unk_id]
                    else:
                        word = self.tgt_vocab.id2token[self.tgt_vocab.unk_id]
                elif word_id == self.tgt_vocab.token2id[UI_BUTTON]:
                    ui_button_id = torch.argmax(ui_button_log_prob[prev_hyp_id])
                    ui_button = schema_list[ui_button_id]
                    word = "_".join(self.plmm_tokenizer.decode(self.plmm_tokenizer.convert_tokens_to_ids(ui_button)).split(' '))
                else:
                    word = self.tgt_vocab.id2token[word_id]

                temp_hyp.tgt_code_tokens.append(word)
                #print(word)
                if word_id == eos_id:
                    temp_hyp.tgt_code_tokens_id = temp_hyp.tgt_code_tokens_id[1:-1]
                    #print(temp_hyp.tgt_code_tokens)
                    temp_hyp.tgt_code_tokens = temp_hyp.tgt_code_tokens[1:-1]
                    temp_hyp.score = new_hyp_score
                    completed_hypotheses.append(temp_hyp)
                else:
                    new_hypotheses.append(temp_hyp)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = new_long_tensor(live_hyp_ids)
            #print(h_t[:,live_hyp_ids].size())
            h_tm1 = (h_t[:,live_hyp_ids], cell_t[:,live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = new_float_tensor(new_hyp_scores)  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses
        if len(completed_hypotheses) == 0:

            #print ("======================= no parsed result !!! =================================")
            #print("======================= no parsed result !!! =================================")
            dummy_hyp = Hypothesis()
            completed_hypotheses.append(dummy_hyp)
        else:
            #completed_hypotheses = [hyp for hyp in completed_hypotheses if hyp.completed()]
            # todo: check the rank order
            completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        params = {
            'state_dict': self.state_dict(),
            'args': self.args,
            'vocab': self.vocab
        }

        torch.save(params, path)

    @classmethod
    def load(cls, model_path, use_cuda=False, loaded_vocab=None, args = None):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']

        saved_args.use_cuda = use_cuda


        parser = cls(vocab, saved_args)

        parser.load_state_dict(saved_state)


        if use_cuda: parser = parser.cuda()

        return parser, None, None
