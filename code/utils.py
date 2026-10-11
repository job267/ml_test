"""Reproducible BPR sampling and ranking metrics."""
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def UniformSample_original(dataset,neg_ratio=1):
    if neg_ratio != 1: raise ValueError('Only 1 negative per positive is supported')
    users=np.random.randint(dataset.n_users,size=dataset.trainDataSize)
    rows=[]
    for u in users:
        positives=dataset.allPos[int(u)]
        if len(positives)==0 or len(positives)>=dataset.m_items: continue
        positive=int(np.random.choice(positives))
        while True:
            negative=int(np.random.randint(dataset.m_items))
            if negative not in positives: break
        rows.append((int(u),positive,negative))
    return np.asarray(rows,dtype=np.int64).reshape(-1,3)

def iter_batches(items,size):
    for start in range(0,len(items),size): yield items[start:start+size]

def metrics_for_user(predicted,ground_truth,k):
    truth=set(ground_truth)
    hits=np.array([int(i in truth) for i in predicted[:k]],dtype=np.float64)
    recall=hits.sum()/len(truth) if truth else 0.
    precision=hits.sum()/k
    discounts=1/np.log2(np.arange(2,k+2))
    dcg=float((hits*discounts).sum())
    ideal=float(discounts[:min(len(truth),k)].sum())
    return precision,recall,dcg/ideal if ideal else 0.

class BPRLoss:
    def __init__(self,model,config):
        self.model=model; self.weight_decay=config['decay']
        self.opt=torch.optim.Adam(model.parameters(),lr=config['lr'])

    def stageOne(self,users,pos,neg):
        loss,reg=self.model.bpr_loss(users,pos,neg)
        total=loss+self.weight_decay*reg
        self.opt.zero_grad(); total.backward(); self.opt.step()
        return float(total.detach())
