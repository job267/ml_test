"""Configuration for the modernized LightGCN baseline."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data'
CHECKPOINT_PATH = ROOT / 'checkpoints'

def make_parser():
    p = argparse.ArgumentParser(description='LightGCN / MF baseline (PyTorch 2.x)')
    p.add_argument('--dataset', choices=['lastfm','gowalla','yelp2018','amazon-book'], default='gowalla')
    p.add_argument('--model', choices=['lgn','mf'], default='lgn')
    p.add_argument('--recdim', type=int, default=64)
    p.add_argument('--layer', type=int, default=3)
    p.add_argument('--bpr_batch', type=int, default=2048)
    p.add_argument('--testbatch', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument('--topks', default='[20]')
    p.add_argument('--tensorboard', type=int, choices=[0,1], default=0)
    p.add_argument('--multicore', type=int, choices=[0,1], default=0)
    p.add_argument('--load', type=int, choices=[0,1], default=0)
    p.add_argument('--pretrain', type=int, choices=[0,1], default=0)
    p.add_argument('--dropout', type=int, choices=[0,1], default=0)
    p.add_argument('--keepprob', type=float, default=0.6)
    p.add_argument('--a_fold', type=int, default=100)
    p.add_argument('--comment', default='lgn')
    p.add_argument('--path', default=str(CHECKPOINT_PATH))
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--data_root', type=Path, default=DATA_PATH)
    p.add_argument('--eval_interval', type=int, default=10)
    return p

def parse_topks(value: str):
    x = ast.literal_eval(value)
    if not isinstance(x,(list,tuple)) or not x or not all(type(v) is int and v > 0 for v in x):
        raise ValueError('--topks must be a nonempty list of positive integers')
    return list(x)

def get_device(choice):
    if choice == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    return torch.device('cuda' if (choice == 'cuda' or (choice == 'auto' and torch.cuda.is_available())) else 'cpu')
