import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from dataloader import Loader
from model import LightGCN,PureMF
from utils import BPRLoss,UniformSample_original
from Procedure import Test
from world import parse_topks

def small_data(tmp_path):
    path=tmp_path/'toy';path.mkdir()
    (path/'train.txt').write_text('0 0 1\n1 1 2\n2 2 3\n')
    (path/'test.txt').write_text('0 2\n1 3\n2 0\n')
    return Loader(path)

def test_graph_symmetry_and_normalization(tmp_path):
    ds=small_data(tmp_path); a=ds.getSparseGraph().to_dense()
    assert a.shape==(7,7)
    assert torch.allclose(a,a.T)
    assert torch.allclose(a[:3,:3],torch.zeros((3,3)))
    assert torch.isfinite(a).all()

def test_model_train_and_eval(tmp_path):
    ds=small_data(tmp_path)
    conf={'latent_dim_rec':8,'lightGCN_n_layers':2,'dropout':0,'keep_prob':0.6,'lr':1e-3,'decay':1e-4}
    for cls in (LightGCN,PureMF):
        net=cls(conf,ds);opt=BPRLoss(net,conf)
        sample=UniformSample_original(ds)
        assert len(sample) > 0
        u,p,n=torch.tensor(sample[:3]).unbind(1)
        assert np.isfinite(opt.stageOne(u,p,n))
        metrics=Test(ds,net,0,testbatch=2,topks=[2])
        assert all(np.isfinite(value).all() for value in metrics.values())

def test_topks_safe():
    assert parse_topks('[20, 40]')==[20,40]
    try: parse_topks('__import__("os").system("echo BAD")')
    except (ValueError,SyntaxError): pass
    else: raise AssertionError('unsafe parsing')
