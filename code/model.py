"""Matrix factorization and the original LightGCN graph aggregation."""
import torch
from torch import nn
from torch.nn import functional as F

class BasicModel(nn.Module):
    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    pass

class PureMF(PairWiseModel):
    def __init__(self,config,dataset):
        super().__init__()
        dim=config['latent_dim_rec']
        self.embedding_user=nn.Embedding(dataset.n_users,dim)
        self.embedding_item=nn.Embedding(dataset.m_items,dim)

    def computer(self):
        return self.embedding_user.weight,self.embedding_item.weight

    def getUsersRating(self,users):
        u,i=self.computer()
        return torch.sigmoid(u[users] @ i.T)

    def bpr_loss(self,users,pos,neg):
        u=self.embedding_user(users); p=self.embedding_item(pos); n=self.embedding_item(neg)
        loss=F.softplus((u*(n-p)).sum(1)).mean()
        reg=(u.square().sum()+p.square().sum()+n.square().sum())/(2*len(users))
        return loss,reg

class LightGCN(PairWiseModel):
    def __init__(self,config,dataset):
        super().__init__()
        self.n_layers=config['lightGCN_n_layers']
        dim=config['latent_dim_rec']
        self.num_users=dataset.n_users
        self.num_items=dataset.m_items
        self.dropout=bool(config['dropout'])
        self.keep_prob=config['keep_prob']
        self.embedding_user=nn.Embedding(self.num_users,dim)
        self.embedding_item=nn.Embedding(self.num_items,dim)
        nn.init.normal_(self.embedding_user.weight,std=0.1)
        nn.init.normal_(self.embedding_item.weight,std=0.1)
        self.register_buffer('Graph',dataset.getSparseGraph(),persistent=False)

    def _dropout_graph(self):
        graph=self.Graph.coalesce()
        if not self.training or not self.dropout:
            return graph
        if not 0 < self.keep_prob <= 1:
            raise ValueError('keep_prob must be in (0,1]')
        keep=torch.rand(graph.values().size(),device=graph.device) < self.keep_prob
        return torch.sparse_coo_tensor(graph.indices()[:,keep],graph.values()[keep]/self.keep_prob,
                graph.shape,device=graph.device).coalesce()

    def computer(self):
        emb=torch.cat((self.embedding_user.weight,self.embedding_item.weight),dim=0)
        outputs=[emb]
        graph=self._dropout_graph()
        for _ in range(self.n_layers):
            emb=torch.sparse.mm(graph,emb)
            outputs.append(emb)
        all_emb=torch.stack(outputs,dim=0).mean(dim=0)
        return torch.split(all_emb,[self.num_users,self.num_items])

    def getUsersRating(self,users):
        u,i=self.computer()
        return torch.sigmoid(u[users] @ i.T)

    def bpr_loss(self,users,pos,neg):
        all_u,all_i=self.computer()
        u=all_u[users]; p=all_i[pos]; n=all_i[neg]
        loss=F.softplus((u*(n-p)).sum(1)).mean()
        u0=self.embedding_user(users); p0=self.embedding_item(pos); n0=self.embedding_item(neg)
        reg=(u0.square().sum()+p0.square().sum()+n0.square().sum())/(2*len(users))
        return loss,reg
