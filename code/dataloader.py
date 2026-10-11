"""User-item interaction dataset and sparse normalized bipartite graph."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

class BasicDataset:
    pass

class Loader(BasicDataset):
    def __init__(self, path, device='cpu'):
        self.path = Path(path)
        self.device = torch.device(device)
        self.trainUser, self.trainItem = self._load_pairs(self.path / 'train.txt')
        self.testUser, self.testItem = self._load_pairs(self.path / 'test.txt')
        if len(self.trainUser) == 0:
            raise ValueError('Training dataset is empty')
        self.n_users = int(max(self.trainUser.max(), self.testUser.max(initial=-1)) + 1)
        self.m_items = int(max(self.trainItem.max(), self.testItem.max(initial=-1)) + 1)
        self.UserItemNet = sp.csr_matrix((np.ones(len(self.trainUser),dtype=np.float32),
            (self.trainUser, self.trainItem)), shape=(self.n_users,self.m_items))
        self.UserItemNet.data[:] = 1.0
        self.UserItemNet.eliminate_zeros()
        self.allPos = [self.UserItemNet[u].indices for u in range(self.n_users)]
        self.trainDataSize = len(self.trainUser)
        self.testDict = {}
        for u,i in zip(self.testUser.tolist(), self.testItem.tolist()):
            self.testDict.setdefault(u, []).append(i)
        self._graph = None

    @staticmethod
    def _load_pairs(file):
        if not file.exists():
            raise FileNotFoundError(f'{file} missing; see README dataset setup')
        users, items = [], []
        for line_no,line in enumerate(file.read_text().splitlines(),1):
            fields = line.split()
            if not fields:
                continue
            u = int(fields[0])
            if u < 0: raise ValueError(f'{file}:{line_no} negative user ID')
            for field in fields[1:]:
                i = int(field)
                if i < 0: raise ValueError(f'{file}:{line_no} negative item ID')
                users.append(u); items.append(i)
        return np.asarray(users,dtype=np.int64), np.asarray(items,dtype=np.int64)

    def getUserPosItems(self, users):
        return [self.allPos[int(u)] for u in users]

    def getSparseGraph(self):
        if self._graph is None:
            r = self.UserItemNet.astype(np.float32)
            adj = sp.bmat([[None,r],[r.T,None]],format='csr',dtype=np.float32)
            deg = np.asarray(adj.sum(axis=1)).ravel()
            inv = np.zeros_like(deg)
            mask = deg > 0
            inv[mask] = np.power(deg[mask],-0.5)
            diagonal = sp.diags(inv)
            normalized = (diagonal @ adj @ diagonal).tocoo()
            indices = torch.tensor(np.vstack((normalized.row,normalized.col)),dtype=torch.long)
            values = torch.tensor(normalized.data,dtype=torch.float32)
            self._graph = torch.sparse_coo_tensor(indices,values,normalized.shape).coalesce().to(self.device)
        return self._graph

class LastFM(Loader):
    """Convert original LastFM pair files into the shared loader format."""
    def __init__(self,path,device='cpu'):
        folder=Path(path)
        for src,dst in [('data1.txt','train.txt'),('test1.txt','test.txt')]:
            if not (folder/dst).exists():
                if not (folder/src).exists(): raise FileNotFoundError(folder/src)
                rows=[]
                by_user={}
                for line in (folder/src).read_text().splitlines():
                    vals=line.split()
                    if len(vals)>=2:
                        u,i=int(vals[0])-1,int(vals[1])-1
                        by_user.setdefault(u,[]).append(i)
                for u,items in by_user.items():
                    rows.append(f'{u} '+ ' '.join(map(str,items)))
                (folder/dst).write_text('\n'.join(rows)+'\n')
        super().__init__(folder,device)
