import pandas as pd
import math
import numpy as np


def Row_Normalization(df):
    n = len(df)
    for i in range(n):
        row_sum = sum(df.iloc[i,:])
        if(math.fabs(row_sum)>0.001):
            df.iloc[i,:] = df.iloc[i,:]/row_sum


def RWR(adj,S,iternum = 2,alpha = 0.5):
    Row_Normalization(S)
    Row_Normalization(adj)
    Init_adj = adj.copy()
    S = S.values
    adj = adj.values
    Init_adj = Init_adj.values
    for i in range(iternum):
        adj = adj.dot(S)
        adj = adj + alpha * Init_adj
    return pd.DataFrame(adj)

def informationFlow(adj,S,alpha = 0.5):
    Row_Normalization(S)
    Row_Normalization(adj)
    S = np.matrix(S.values)
    adj = np.matrix(adj.values)
    I = np.identity(len(S))
    ret = adj * (I - alpha* S).I
    return ret




def Save(df,savename):
    path =  "../Data/Features/" + savename
    df.to_csv(path,sep="\t")


if __name__ == '__main__':
    DG = pd.read_csv("../Data/Disease/DG.txt",sep='\t',index_col=0)
    GG = pd.read_csv("../Data/gene/GG.txt",sep='\t',index_col=0)
    # MG = pd.read_csv("../Data/miRNA/MG.txt",sep='\t',index_col=0)
    DG = informationFlow(DG,GG)
    print(np.max(DG))
    print(np.min(DG))
    print(DG[0,:])
    # MG = RWR(MG,GG,50)
    # print(MG.iloc[0,:])
    # Save(DG,"DiseaseFeature.txt")
    # Save(MG,"MiRNAFeature.txt")
