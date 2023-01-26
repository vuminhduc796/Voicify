# coding=utf-8
from __future__ import print_function

import sys
import traceback

import numpy as np

from tqdm import tqdm


def decode(examples, model, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    decode_results = []
    if args.batch_decode:
        batch_size = args.batch_size
        index_arr = np.arange(len(examples))
        batch_num = int(np.ceil(len(examples) / float(batch_size)))
        for batch_id in tqdm(range(batch_num)):
            # print (batch_id)
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [examples[i] for i in batch_ids]
            hyps = model.batch_decode(batch_examples, args.decode_max_time_step)
            decode_results.extend([[hyp] for hyp in hyps])
    else:
        for example in examples:
            if args.parser == "seq2seq_c_t":
                hyps = model.beam_search(example, args.decode_max_time_step, beam_size=args.beam_size, relax_factor=args.relax_factor)
            elif args.parser == 'pretrain_seq2seq':
                hyps = model.beam_search(example, kwargs['domain'], kwargs['src_lang'], kwargs['tgt_lang'], args.decode_max_time_step, beam_size=args.beam_size)
            else:
                hyps = model.beam_search(example, args.decode_max_time_step, beam_size=args.beam_size)

            decode_results.append(hyps)

    if was_training: model.train()

    return decode_results


def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=True, **kwargs):
    decode_results = decode(examples, parser, args, verbose=verbose, **kwargs)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only)

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
