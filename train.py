#!/usr/bin/env python
import argparse
import multiprocessing as mp

import torch

from agent.training import Training
from agent.utils import populate_config

if __name__ == '__main__':
    print('Version 1.0')
    torch.set_num_threads(1)
    print(torch.get_num_threads())
    mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.')
    parser.add_argument('--entropy_beta', type=float, default=0.01,
                        help='entropy beta (default: 0.01)')

    parser.add_argument('--restore', action='store_true',
                        help='restore from checkpoint')
    parser.add_argument('--grad_norm', type=float, default=40.0,
                        help='gradient norm clip (default: 40.0)')

    parser.add_argument('--h5_file_path', type=str,
                        default='./data/{scene}.h5')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/model/checkpoint-{checkpoint}.pth')

    parser.add_argument('--learning_rate', type=float,
                        default=0.0007001643593729748)
    parser.add_argument('--rmsp_alpha', type=float, default=0.99,
                        help='decay parameter for RMSProp optimizer (default: 0.99)')
    parser.add_argument('--rmsp_epsilon', type=float, default=0.1,
                        help='epsilon parameter for RMSProp optimizer (default: 0.1)')

    # Use experiment.json
    parser.add_argument('--exp', '-e', type=str,
                        help='Experiment parameters.json file', required=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Disable OMP
    torch.set_num_threads(1)

    args = vars(parser.parse_args())
    args = populate_config(args)

    if args.get('method', None) is None:
        print('ERROR Please choose a method in json file')
        print('- "ana"')
        print('- "aop"')
        print('- "word2vec"')
        print('- "target_driven"')

        exit()

    torch.manual_seed(args['seed'])

    if args['restore']:
        t = Training.load_checkpoint(args)
    else:
        t = Training(args)

    t.run()
