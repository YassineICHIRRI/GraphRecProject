import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy import sparse
import numpy as np

class LightGCN(Module):

    def __init__(self, userNum, itemNum, rt, embedSize=100, numLayers=3):
        super(LightGCN, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.embedSize = embedSize
        self.numLayers = numLayers

        # Embedding layers
        self.uEmbd = nn.Embedding(userNum, embedSize)
        self.iEmbd = nn.Embedding(itemNum, embedSize)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.uEmbd.weight)
        nn.init.xavier_uniform_(self.iEmbd.weight)

        # Build adjacency matrix
        self.LaplacianMat = self.buildLaplacianMat(rt)

    def buildLaplacianMat(self, rt):
        rt_item = rt['movie_id_ml'] + self.userNum
        uiMat = coo_matrix((rt['rating'], (rt['user_id'], rt['movie_id_ml'])))

        uiMat_upperPart = coo_matrix((rt['rating'], (rt['user_id'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))

        A = sparse.vstack([uiMat_upperPart, uiMat])
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr).flatten()
        diag = np.power(diag, -0.5, where=diag != 0)
        D = sparse.diags(diag)
        L = D @ A @ D

        L = sparse.coo_matrix(L)
        row, col = L.row, L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data)

        return SparseL

    def getFeatureMat(self):
        uidx = torch.arange(self.userNum)
        iidx = torch.arange(self.itemNum)

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd, itemEmbd], dim=0)

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
