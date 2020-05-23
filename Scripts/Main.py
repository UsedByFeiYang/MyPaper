import pandas as pd
import DataPreprocess
from DataPreprocess import DataReader
import torch as t
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits



class Myloss(nn.Module):

    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, y_true, y_hat,m_x0,m_x1,m_x2,m_x3,d_x0, d_x1, d_x2,d_x3):
        loss = binary_cross_entropy_with_logits(y_hat,y_true,reduction='mean')
        origin_loss = t.norm(m_x0 - m_x1) + t.norm(m_x0 - m_x1) + t.norm(m_x0 - m_x2) +t.norm(m_x0 - m_x2) + t.norm(d_x0 - d_x1) + t.norm(d_x0 - d_x2) + t.norm(d_x0 - d_x3)
        inter_loss = t.norm(m_x1 - m_x2) + t.norm(m_x1 - m_x3) + t.norm(m_x2 - m_x3) + t.norm(d_x1 - d_x2) + t.norm(d_x1 - d_x3) + t.norm(d_x2 - d_x3)
        return loss + origin_loss -inter_loss




class Train():
    def __init__(self,MD,MSS,MSG,MSF,MF,DSP,DSSE,DSSP,DF):
        self.MD = MD

        self.MSS = MSS
        self.MSG = MSG
        self.MSF = MSF
        self.MF = MF

        self.DSP = DSP
        self.DSSE = DSSE
        self.DSSP = DSSP
        self.DF = DF


    @staticmethod
    def train(model, train_data, optimizer,epoch):
        model.train()
        crit = Myloss()
        one_index = train_data[2][0].t().tolist()
        zero_index = train_data[2][1].t().tolist()

    def train_epoch(self,model,train_data,crit):
        model.zero_grad()
        score = model(train_data)
        loss = crit(one_index, zero_index, train_data[4], score)
        loss.backward()
        optimizer.step()
        return loss

        for epoch in range(epoch):
            train_reg_loss = train_epoch()
            print(train_reg_loss.item() / (len(one_index[0]) + len(zero_index[0])))

class Test():
    @staticmethod
    def test():
        pass



if __name__ == '__main__':
    # load data of miRNA
    MSS = DataReader.read_excel("../Data/miRNA/MS_Sequence.xlsx")
    MSG = DataReader.read_excel("../Data/miRNA/MS_GO.xlsx")
    MSF = DataReader.read_excel("../Data/miRNA/MS_Function.xlsx")
    M_F = DataReader.read_excel("../Data/Features/miRNAFeatureCompressed.xlsx")

    #load data of disease
    DSP = DataReader.read_excel("../Data/Disease/DS_Phenotype.xlsx")
    DSSE = DataReader.read_excel("../Data/Disease/DS_Semantic.xlsx")
    DSSY = DataReader.read_excel("../Data/Disease/DS_Symptom.xlsx")
    D_F = DataReader.read_excel("../Data/Features/DiseaseFeatureCompressed.xlsx")

    #load data of miRNA-disease association
    M_D = DataReader.read_excel("../Data/MD.xlsx")

