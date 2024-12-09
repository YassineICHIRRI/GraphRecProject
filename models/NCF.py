import torch
import torch.nn as nn 
from torch.nn import Module 


class NCF(Module):

    def __init__(self, userNum, itemNum, dim=64, first_layer=128):
        super(NCF, self).__init__()

        self.uEmbd = nn.Embedding(userNum, dim)
        self.iEmbd = nn.Embedding(itemNum, dim)

        self.mf_uEmbd = nn.Embedding(userNum, dim)
        self.mf_iEmbd = nn.Embedding(itemNum, dim)

        self.mlp = nn.Sequential(nn.Linear(dim*2, first_layer),
                                       nn.Dropout(0.25),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(first_layer, first_layer//2),
                                       nn.Dropout(0.25),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(first_layer//2, first_layer//4),
                                       nn.Dropout(0.25),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(first_layer//4, first_layer//4))

        self.neumf = nn.Linear(dim+first_layer//4, 1)

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        embd = torch.cat([uembd, iembd], dim=1)

        mlp = self.mlp(embd)
        mf = self.uEmbd(userIdx)*self.iEmbd(itemIdx)

        prediction = self.neumf(torch.cat([mlp, mf], dim=1))

        return prediction.flatten()