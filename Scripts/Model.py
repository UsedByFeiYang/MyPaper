import torch as t
from  torch.nn.functional import  softmax
from torch import nn
from torch_geometric.nn import conv
from torch_geometric.utils import dropout_adj

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel,self).__init__()
        self.cnn1 = t.nn.Conv2d(in_channels= 1 , out_channels= 2,kernel_size=(1,3),padding=(0,2))
        self.cnn2 = t.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3))
    def forward(self, X):
        X = X.view(1, 1, X.size(0), X.size(1))
        X = self.cnn1(X)
        X = t.relu(X)
        X = self.cnn2(X)
        X = t.relu(X)
        return X.view(X.size(2), X.size(3))


class FullConnectedLayer(nn.Module):
    def __init__(self,featureLen):
        super(FullConnectedLayer,self).__init__()
        self.l1 = nn.Linear(featureLen,256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)

        self.dropout1 = nn.Dropout(p=0)
        self.dropout2 = nn.Dropout(p = 0)

    def forward(self, input):

        x = t.relu(self.l1(input))
        x = self.dropout1(x)
        x = t.relu(self.l2(x))
        x = self.dropout2(x)
        x = t.relu(self.l3(x))
        return x


class AttentionLayer(nn.Module):
    def __init__(self,featurelen):
        super(AttentionLayer,self).__init__()
        self.l0 = nn.Linear(featurelen, 16)
        self.l1 = nn.Linear(featurelen,16)
        self.l2 = nn.Linear(featurelen, 16)
        self.l3 = nn.Linear(featurelen, 16)
        self.ls = nn.Linear(16, 64)
    def forward(self,X0,X1,X2,X3):
        X0 = self.l0(X0)
        X1 = self.l1(X1)
        X2 = self.l2(X2)
        X3 = self.l3(X3)
        Au = self.ls(X0 + X1 + X2 + X3)
        Au = softmax(Au,dim=1)
        X = t.cat((X0,X1,X2,X3),dim=1)
        X = t.mul(X,Au)
        return X




class SingleModule(nn.Module):
    def __init__(self,featureLen):
        super(SingleModule,self).__init__()

        self.gcn1 = conv.GCNConv(featureLen,featureLen)
        self.gcn2 = conv.GCNConv(featureLen,featureLen)

    def forward(self, X,adj):

        edge_index,edge_weight = dropout_adj(adj.data['edge'].cuda(),adj.data['weight'][adj.data['edge'][0],adj.data['edge'][1]].cuda(),p=0.2)
        X = self.gcn1(X.cuda(), edge_index,edge_weight)

        # X = self.gcn1(X.cuda(),adj.data['edge'].cuda(),adj.data['weight'][adj.data['edge'][0],adj.data['edge'][1]].cuda())
        X = t.relu(X)


        X = self.gcn2(X.cuda(), edge_index, edge_weight)
        #X = self.gcn2(X.cuda(), adj.data['edge'].cuda(), adj.data['weight'][adj.data['edge'][0].cuda(), adj.data['edge'][1]].cuda())
        X = t.relu(X)

        return X

class SubModel(nn.Module):
    def __init__(self,X,graph1,graph2,graph3):
        super(SubModel,self).__init__()

        #data
        self.graph1 = graph1
        self.graph2 = graph2
        self.graph3 = graph3
        self.X = X.cuda()

        #model
        self.model1 = SingleModule(self.X.size(1))
        self.model2 = SingleModule(self.X.size(1))
        self.model3 = SingleModule(self.X.size(1))
        self.linear1 = FullConnectedLayer(self.X.size(1)) #skip connection
        self.linear2 = FullConnectedLayer(self.X.size(1)) #siamise MLP

        #information fusion
        #self.attention = AttentionLayer(64)



    def forward(self):
        X0 = self.linear1(self.X)

        X1 = self.model1(self.X,self.graph1)
        X1 = self.linear2(X1)


        X2 = self.model2(self.X, self.graph2)
        X2 = self.linear2(X2)

        X3 = self.model3(self.X, self.graph3)
        X3 = self.linear2(X3)


        #information fusion
        X = X0 + X1 + X2 + X3       # method 1 : element-wise sum

        #X = t.cat((X0,X1,X2,X3),1)   # method 2 :concation

        #X = self.attention(X0,X1,X2,X3)  #method3: attention
        #return X,X0,X1,X2,X3
        return X, X1, X2, X3



class MyModel(nn.Module):
    def __init__(self,disease_feature,disease_graph1,disease_graph2,disease_graph3,miRNA_feature,miRNA_graph1,miRNA_graph2,miRNA_graph3):
        super(MyModel,self).__init__()

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
        m_x,m_x1,m_x2,m_x3 = self.miRNA_model()
        d_x, d_x1, d_x2, d_x3 = self.disease_model()
        Y_hat = t.mm(m_x,d_x.T)
        return Y_hat,m_x,m_x1,m_x2,m_x3,d_x, d_x1, d_x2,d_x3






