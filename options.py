import argparse
from random import seed
import os

def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--output_path', type = str, default = 'outputs/')
    parser.add_argument('--root_dir', type = str, default = 'outputs/')
    parser.add_argument('--log_path', type = str, default = 'logs/')
    parser.add_argument('--modal', type = str, default = 'rgb',choices = ["rgb,flow,both"])
    parser.add_argument('--model_path', type = str, default = 'models/')
    parser.add_argument('--lr', type = str, default = '[0.0005]*3000', help = 'learning rates for steps(list form)')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--num_segments', type = int, default = 32)
    parser.add_argument('--seed', type = int, default = 2022, help = 'random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type = str, default = "trans_{}.pkl".format(seed), help = 'the path of pre-trained model file')
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--num-classes', type=int, default=1, help='number of class')


    parser.add_argument('--dropout', default=0.6, help='x x')
    parser.add_argument('--model', default='HGCN', help='which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN, HyboNet]')
    parser.add_argument('--manifold', default='PoincareBall', help='which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall, Lorentz]')
    parser.add_argument('--c', default=None, help='hyperbolic radius, set to None for trainable curvature')
    parser.add_argument('--num-layers', default=2, help='layers of hgcn')
    parser.add_argument('--act', default='relu', help='which activation function to use (or None for no activation)')
    parser.add_argument('--bias', default=1, help='whether to use bias (1) or not (0)')
    parser.add_argument('--use-att', default=0, help='whether to use hyperbolic attention or not')
    parser.add_argument('--local-agg', default=0, help='whether to local tangent space aggregation or not')
    parser.add_argument('--tie_weight', default=True, help='whether to tie transformation matrices')
    parser.add_argument('--gcn-in-feat', type=int, default=128, help='input size of feature for HGCN (default: 2048)')
    parser.add_argument('--gcn-hidden', type=int, default=32, help='embedding dimension')
    parser.add_argument('--c-in-feat', type=int, default=1024, help='input size of feature for HGCN (default: 2048)')
    parser.add_argument('--c-hidden', type=int, default=512, help='embedding dimension')
    parser.add_argument('--c-out-feat', type=int, default=128, help='input size of feature for HGCN (default: 2048)')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args
