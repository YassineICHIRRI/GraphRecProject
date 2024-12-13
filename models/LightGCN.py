import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy import sparse
import numpy as np

class LightGCN(Module):

    def __init__(self, userNum, itemNum, rt, embedSize=100, numLayers=3,useCuda = False):
        super(LightGCN, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.embedSize = embedSize
        self.numLayers = numLayers
        self.useCuda = useCuda
        # Embedding layers
        self.uEmbd = nn.Embedding(userNum, embedSize)
        self.iEmbd = nn.Embedding(itemNum, embedSize)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.uEmbd.weight)
        nn.init.xavier_uniform_(self.iEmbd.weight)
        self.LaplacianMat = self.buildLaplacianMat(rt) # sparse format
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)

    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def buildLaplacianMat(self,rt):

        rt_item = rt['movie_id_ml'] + self.userNum
        uiMat = coo_matrix((rt['rating'], (rt['user_id'], rt['movie_id_ml'])))

        uiMat_upperPart = coo_matrix((rt['rating'], (rt['user_id'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.userNum+self.itemNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])

        device = torch.device("cuda" if torch.cuda.is_available() and self.useCuda else "cpu")
        uidx = uidx.to(device)
        iidx = iidx.to(device)


        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def propagate(self, features):
        all_embeds = [features]

        for _ in range(self.numLayers):
            features = torch.sparse.mm(self.LaplacianMat, features)
            all_embeds.append(features)

        return torch.mean(torch.stack(all_embeds, dim=0), dim=0)

    def forward(self, userIdx, itemIdx):
        features = self.getFeatureMat()

        # Propagate embeddings
        finalEmbd = self.propagate(features)

        # Separate user and item embeddings
        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx + self.userNum]

        # Dot product for prediction
        prediction = torch.sum(userEmbd * itemEmbd, dim=1)

        return prediction
