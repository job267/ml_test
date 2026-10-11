"""BPR train/evaluate procedures, preserving the original objective."""
import numpy as np
import torch
from utils import UniformSample_original,iter_batches,metrics_for_user

def BPR_train_original(dataset,recommend_model,loss_class,epoch,neg_k=1,w=None,batch_size=2048):
    recommend_model.train()
    triplets=UniformSample_original(dataset,neg_k)
    np.random.shuffle(triplets)
    device=next(recommend_model.parameters()).device
    total=0.; batches=0
    for batch in iter_batches(triplets,batch_size):
        u,p,n=torch.as_tensor(batch,device=device,dtype=torch.long).unbind(1)
        total+=loss_class.stageOne(u,p,n); batches+=1
    value=total/max(batches,1)
    if w: w.add_scalar('train/bpr',value,epoch)
    return {'loss':value,'samples':len(triplets)}

@torch.no_grad()
def Test(dataset,Recmodel,epoch,w=None,multicore=0,testbatch=100,topks=(20,)):
    Recmodel.eval()
    device=next(Recmodel.parameters()).device
    users,items=Recmodel.computer()  # propagate ONCE for all evaluation batches
    maxk=max(topks)
    if maxk>dataset.m_items: raise ValueError('top-k larger than total items')
    sums=np.zeros((len(topks),3),dtype=np.float64)
    evaluated=0
    ids=list(dataset.testDict)
    for subset in iter_batches(ids,testbatch):
        u=torch.tensor(subset,device=device,dtype=torch.long)
        scores=users[u] @ items.T
        for row,user in enumerate(subset):
            positives=dataset.allPos[user]
            scores[row,torch.as_tensor(positives,device=device,dtype=torch.long)]=-float('inf')
        ranks=torch.topk(scores,maxk,dim=1).indices.cpu().numpy()
        for row,user in enumerate(subset):
            for index,k in enumerate(topks):
                sums[index]+=metrics_for_user(ranks[row],dataset.testDict[user],k)
            evaluated+=1
    sums/=max(evaluated,1)
    result={'precision':sums[:,0],'recall':sums[:,1],'ndcg':sums[:,2]}
    if w:
        for i,k in enumerate(topks):
            for name,values in result.items():w.add_scalar(f'test/{name}@{k}',values[i],epoch)
    return result
