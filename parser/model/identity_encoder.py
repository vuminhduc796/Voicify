#
# Copyright (c) 2020 The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn

from .common import LayerNorm, LinearFeedforward


class IdentityEncoder(nn.Module):
    def __init__(self, src_vocab, args, config, context_embeddings):
        super().__init__()
        self.args = args
        self.pad_idx = src_vocab.pad_id

        if args.hidden_size is None:
            args.hidden_size = config.hidden_size

        self.encoder_embeddings = context_embeddings

        self.projection = None
        if self.args.decoder_layer_size > 0 and self.args.hidden_size != config.hidden_size:
            self.dropout = nn.Dropout(args.dropout)
            self.projection = nn.Linear(config.hidden_size, self.args.hidden_size, bias=False)

        if self.args.decoder_layer_size > 0 and self.args.rnn_zero_state in ['average', 'cls']:
            self.pool = LinearFeedforward(
                args.hidden_size, args.hidden_size, 2 * args.hidden_size * args.decoder_layer_size, dropout=args.dropout
            )
            self.norm = LayerNorm(2 * args.hidden_size * args.decoder_layer_size)

    def compute_final_embeddings(
        self,
        context,
        context_lengths,
        context_padding,
    ):
        #print(context.size())
        #print(context_lengths)

        context_embedded_last_hidden_state = self.encoder_embeddings(
            context, attention_mask=(~context_padding).to(dtype=torch.float)
        ).last_hidden_state

        final_context = context_embedded_last_hidden_state

        if self.projection is not None:
            final_context = self.dropout(final_context)
            final_context = self.projection(final_context)

        context_rnn_state = None
        if self.args.decoder_layer_size > 0:
            batch_size = context.size(0)
            if self.args.rnn_zero_state == 'zero':

                zero = torch.zeros(
                    self.args.decoder_layer_size,
                    batch_size,
                    self.args.rnn_dimension,
                    dtype=torch.float,
                    requires_grad=False,
                    device=context.device,
                )
                context_rnn_state = (zero, zero)
            else:
                if self.args.rnn_zero_state == 'cls':
                    packed_rnn_state = self.norm(self.pool(final_context[:, 0, :]))

                else:
                    assert self.args.rnn_zero_state == 'average'
                    masked_final_context = final_context.masked_fill(context_padding.unsqueeze(2), 0)
                    summed_context = torch.sum(masked_final_context, dim=1)
                    #print(context_lengths.unsqueeze(1).size())
                    #print(summed_context.size())
                    average_context = summed_context / context_lengths.unsqueeze(1)

                    #print(average_context.size())

                    packed_rnn_state = self.norm(self.pool(average_context))

                # packed_rnn_state is (batch, 2 * rnn_layers * rnn_dim)
                packed_rnn_state = packed_rnn_state.reshape(batch_size, 2, self.args.decoder_layer_size, self.args.hidden_size)
                # transpose to (2, batch, rnn_layers, rnn_dimension)
                packed_rnn_state = packed_rnn_state.transpose(0, 1)
                # transpose to (2, rnn_layers, batch, rnn_dimension)
                packed_rnn_state = packed_rnn_state.transpose(1, 2)
                # convert to a tuple of two (rnn_layers, batch, rnn_dimension) tensors
                packed_rnn_state = packed_rnn_state.chunk(2, dim=0)
                context_rnn_state = (packed_rnn_state[0].squeeze(0), packed_rnn_state[1].squeeze(0))

        #print(context_rnn_state[0].size())

        return final_context, context_rnn_state

    def forward(self, input_ids, input_lengths):

        context_padding = torch.eq(input_ids, self.pad_idx)

        #print(context_padding)

        final_context, context_rnn_state = self.compute_final_embeddings(
            input_ids,
            input_lengths,
            context_padding
        )

        return final_context, context_rnn_state
