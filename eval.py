#!/usr/bin/env python
import argparse
import multiprocessing as mp
import sys

import torch

from agent.evaluation import Evaluation
from agent.evaluation_savn import Evaluation_savn
# from agent.evaluation_att import Evaluation
from agent.utils import populate_config

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.')
    parser.add_argument('--h5_file_path', type=str,
                        default='./data/{scene}.h5')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval_type', type=str, default='val_known')

    # Use experiment.json
    parser.add_argument('--exp', '-e', type=str,
                        help='Experiment parameters.json file', required=True)

    args = vars(parser.parse_args())

    if args['checkpoint_path'] is not None:
        if args['train']:
            args = populate_config(args, mode='train', checkpoint=False)
        else:
            args = populate_config(args, mode='eval', checkpoint=False)
    else:
        if args['train']:
            args = populate_config(args, mode='train')
        else:
            args = populate_config(args, mode='eval')

    if args.get('method', None) is None:
        print('ERROR Please choose a method in json file')
        print('- "aop"')
        print('- "word2vec"')
        print('- "word2vec_noconv"')
        print('- "word2vec_nosimi"')
        print('- "target_driven"')
        print('- "random"')

        exit()

    if args.eval_type == 'test_known':
        t = Evaluation_savn.load_checkpoints(args)
    else:
        t = Evaluation.load_checkpoints(args)

    t.run(args['show'])
