"""Training entrypoint. Run from repository root: python code/main.py ..."""
from pathlib import Path
import sys
import torch
from world import make_parser,parse_topks,get_device,CHECKPOINT_PATH
from dataloader import Loader,LastFM
from model import LightGCN,PureMF
from utils import set_seed,BPRLoss
from Procedure import Test,BPR_train_original

def run(argv=None):
    args=make_parser().parse_args(argv)
    if args.bpr_batch<1 or args.testbatch<1 or args.eval_interval<1 or args.layer<0:
        raise ValueError('batch size/evaluation interval must be positive and layers nonnegative')
    topks=parse_topks(args.topks)
    device=get_device(args.device)
    set_seed(args.seed)
    path=args.data_root/args.dataset
    dataset=(LastFM if args.dataset=='lastfm' else Loader)(path,device)
    config={'latent_dim_rec':args.recdim,'lightGCN_n_layers':args.layer,'dropout':args.dropout,
            'keep_prob':args.keepprob,'lr':args.lr,'decay':args.decay}
    recmodel=(LightGCN if args.model=='lgn' else PureMF)(config,dataset).to(device)
    optimizer=BPRLoss(recmodel,config)
    path=Path(args.path)
    path.mkdir(parents=True,exist_ok=True)
    output=path/f'{args.model}-{args.dataset}-{args.layer}-{args.recdim}.pth.tar'
    if args.pretrain:
        raise ValueError('Legacy --pretrain=1 expects missing embedding arrays; use --load=1 for checkpoints')
    if args.load:
        if not output.exists(): raise FileNotFoundError(output)
        recmodel.load_state_dict(torch.load(output,map_location=device,weights_only=True))
    writer=None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(comment=args.comment)
    try:
        for epoch in range(args.epochs):
            if epoch%args.eval_interval==0:
                result=Test(dataset,recmodel,epoch,w=writer,multicore=args.multicore,
                            testbatch=args.testbatch,topks=topks)
                print('TEST',epoch,{k:v.tolist() for k,v in result.items()},flush=True)
            stats=BPR_train_original(dataset,recmodel,optimizer,epoch,w=writer,batch_size=args.bpr_batch)
            print(f'EPOCH {epoch+1}/{args.epochs}',stats,flush=True)
            torch.save(recmodel.state_dict(),output)
    finally:
        if writer: writer.close()
    return recmodel

if __name__=='__main__':run()
