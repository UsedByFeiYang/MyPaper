import torch as t
from torch import nn
from torch_geometric.nn import conv


class SingleModule(nn.Module):
    def __init__(self,featureLen):
        super(SingleModule,self).__init__()

        self.gcn1 = conv.GCNConv(featureLen,featureLen)
        self.gcn2 = conv.GCNConv(featureLen,featureLen)
        self.cnn1  = t.nn.Conv2d(in_channels= 1 , out_channels= 2,kernel_size=(1,3))
        self.cnn2 = t.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3))

    def forward(self, X,adj):
        X = self.gcn1(X,adj.data['edge'],adj.data['weight'][adj.data['edge'][0],adj.data['edge'][1]])
        X = t.relu(X)
        X = self.gcn2(X, adj.data['edge'], adj.data['weight'][adj.data['edge'][0], adj.data['edge'][1]])
        X = t.relu(X)
        X = X.view(1,1,X.size(0),X.size(1))
        X = self.cnn1(X)
        X = t.relu(X)
        X = self.cnn2(X)
        X = t.relu(X)
        return X.view(X.size(2),X.size(3))

class SubModel(nn.Module):
    def __init__(self,X,graph1,graph2,graph3):
        super(SubModel,self).__init__()

        #data
        self.graph1 = graph1
        self.graph2 = graph2
        self.graph3 = graph3
        self.X = X

        #model
        self.model1 = SingleModule(self.X.size(1))
        self.model2 = SingleModule(self.X.size(1))
        self.model3 = SingleModule(self.X.size(1))

        self.origin1 = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 3))
        self.origin2 = t.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3))


    def forward(self):
        X0 = self.origin1(self.X)
        X0 = t.relu(X0)
        X0 = self.origin2(X0)
        X0 = t.relu(X0)


        X1 = self.GCN1(self.X,self.graph1)
        X2 = self.GCN2(self.X, self.graph2)
        X3 = self.GCN3(self.X, self.graph3)
        X = X0 + X1 + X2 + X3

        return X,X0,X1,X2,X3



class Model(nn.Module):
    def __init__(self,disease_feature,disease_graph1,disease_graph2,disease_graph3,miRNA_feature,miRNA_graph1,miRNA_graph2,miRNA_graph3):
        super(Model,self).__init__()

        self.disease_feature = disease_feature
        self.disease_graph1 = disease_graph1
        self.disease_graph2 = disease_graph2
        self.disease_graph3 = disease_graph3

        self.miRNA_feature = miRNA_feature
        self.miRNA_graph1 = miRNA_graph1
        self.miRNA_graph2 = miRNA_graph2
        self.miRNA_graph3 = miRNA_graph3


        self.disease_model = SubModel(self.disease_feature,self.disease_graph1,self.disease_graph2,self.disease_graph3)
        self.miRNA_model   = SubModel(self.miRNA_feature,self.miRNA_graph1,self.miRNA_graph2,self.miRNA_graph3)

    def forward(self):
        m_x,m_x0,m_x1,m_x2,m_x3 = self.miRNA_model()
        d_x,d_x0, d_x1, d_x2, d_x3 = self.miRNA_model()
        Y_hat = t.mm(m_x,d_x.T)
        return Y_hat,m_x0,m_x1,m_x2,m_x3,d_x0, d_x1, d_x2,d_x3






