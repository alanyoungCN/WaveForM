import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--model_id", type=str, default="ECL_96_96", help="model id")
parser.add_argument(
    "--model",
    type=str,
    default="test",
    help="model name, options: [WaveForM]",
)

parser.add_argument("--data", type=str, default="custom", help="dataset type")
parser.add_argument(
    "--root_path",
    type=str,
    default="./dataset/electricity/",
    help="root path of the data file",
)
parser.add_argument(
    "--data_path", type=str, default="electricity.csv", help="data file"
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)

parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
parser.add_argument(
    "--pred_len", type=int, default=96, help="prediction sequence length"
)

parser.add_argument("--n_points", type=int, default=321, help="the number of variables")
parser.add_argument("--dropout", type=float, default=0.05, help="dropout")

parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="Exp", help="exp description")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")

parser.add_argument("--node_dim", type=int, default=40, help="node_dim in graph")
parser.add_argument("--subgraph_size", type=int, default=6, help="the subgraph size, i.e. topk")
parser.add_argument("--n_gnn_layer", type=int, default=3, help="number of layers in GNN.")
parser.add_argument("--wavelet_j", type=int, default=2, help="the number of wavelet layer")
parser.add_argument("--wavelet", type=str, default='haar', help='the wavelet function')

args = parser.parse_args()


print("Args in experiment:")
import json

print(json.dumps(vars(args), indent=4, ensure_ascii=False))

Exp = Exp_Main

from utils import color
for ii in range(args.itr):
    setting = f"{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_{args.des}_{ii}"
    
    exp = Exp(args)
    color.cprint(f'start training:\n{setting}', color.OKGREEN, end='\n')
    
    exp.train(setting)
    
    color.cprint(f'end of training. begin testing', color.OKGREEN, '\n')
    exp.test(setting)
