import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy import sparse
import numpy as np

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNLayer(Module):

    def __init__(self, inF, outF):

        super(GNNLayer, self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF, out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF, out_features=outF)

    def forward(self, laplacianMat, selfLoop, features):
        L1 = laplacianMat + selfLoop
        L1 = L1.to(device)
        L2 = laplacianMat.to(device)
        inter_feature = torch.mul(features, features)

        inter_part1 = self.linear(torch.sparse.mm(L1, features))
        inter_part2 = self.interActTransform(torch.sparse.mm(L2, inter_feature))

        return inter_part1 + inter_part2

class NGCF(Module):

    def __init__(self, userNum, itemNum, rt, embedSize=100, layers=[100, 80, 50], useCuda=True):
        super(NGCF, self).__init__()
        self.device = torch.device("cuda" if useCuda and torch.cuda.is_available() else "cpu")
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum, embedSize)
        self.iEmbd = nn.Embedding(itemNum, embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        self.LaplacianMat = self.buildLaplacianMat(rt)  # sparse format
        self.selfLoop = self.getSparseEye(self.userNum + self.itemNum)

        # Correct calculation of final_feature_size
        gnn_output_size = sum(layers) + embedSize  # Embed size + output of all GNN layers
        self.transForm1 = nn.Linear(in_features=gnn_output_size * 2, out_features=64)
        self.transForm2 = nn.Linear(in_features=64, out_features=32)
        self.transForm3 = nn.Linear(in_features=32, out_features=1)

        # Initialize GNN layers
        for From, To in zip([embedSize] + layers[:-1], layers):
            self.GNNlayers.append(GNNLayer(From, To))

    def getSparseEye(self, num):
        # Create a sparse identity matrix
        i = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    def buildLaplacianMat(self, rt):
        # Build the Laplacian matrix from the user-item interaction data
        rt_item = rt['movie_id_ml'] + self.userNum
        uiMat = coo_matrix((rt['rating'], (rt['user_id'], rt['movie_id_ml'])))

        uiMat_upperPart = coo_matrix((rt['rating'], (rt['user_id'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))

        A = sparse.vstack([uiMat_upperPart, uiMat])
        selfLoop = sparse.eye(self.userNum + self.itemNum)
        sumArr = (A > 0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag, -0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)

        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data)
        return SparseL

    def getFeatureMat(self):
        # Generate the feature matrix by concatenating user and item embeddings
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd, itemEmbd], dim=0)
        return features

    def forward(self, userIdx, itemIdx):
        itemIdx = itemIdx + self.userNum
        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)

        # GCF data propagation
        features = self.getFeatureMat()
        finalEmbd = features.clone()

        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat, self.selfLoop, features)
            features = nn.LeakyReLU()(features)
            finalEmbd = torch.cat([finalEmbd, features.clone()], dim=1)

        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        embd = torch.cat([userEmbd, itemEmbd], dim=1)

        # Pass through fully connected layers
        embd = nn.ReLU()(self.transForm1(embd))
        embd = nn.ReLU()(self.transForm2(embd))
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction
