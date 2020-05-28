import torch as t
from torch_geometric.data import  Data
from torch import nn



# conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,2),bias=False,padding=(0,1))
# input = t.randn(2,2)
# input = input.view(1,1,2,2)
# # # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
# # #input = input.permute(0,2,1)
# print("---------input:--------")
# print(input)
# out = conv1(input)
# print("---------outsize:--------")
# print(out.size())
# print(out)
# # #out = out.view(2,2)
# # print("---------out:--------")
# # print(out)
# # print("---------conv1:--------")
# print(conv1.state_dict())
#
# conv2 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(1,2),bias=False)
# out1 = conv2(out)
# print("---------out1:--------")
# print(out1)
# print("---------conv2:--------")
# print(conv2.state_dict())

# conv1 = nn.Conv1d(in_channels=256,out_channels=256,kernel_siz
# e=2)
# input = t.randn(1,256,35)
# # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
# #input = input.permute(0,2,1)
# out = conv1(input)
# print(out.size())



# edge_index = t.tensor([[0,2,1,3],[1,3,0,2]],dtype= t.long)
# # x = t.tensor([[-1],[0],[1],[2],[3]],dtype= t.float)
# data = Data(edge_index= edge_index)
# print(data)
# print(data.num_edges)
# print(data.contains_isolated_nodes())
# edge_index = t.tensor([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]],dtype= t.long)
# x = t.tensor([[-1],[0],[1],[2],[3]],dtype= t.float)
# data = Data(x = x,edge_index= edge_index.t().contiguous())
# print(data)
# print(data.x)
# print(data.num_edges)
# print(data.num_nodes)
# print(data.contains_isolated_nodes())

# print("transform before")
# x = t.randn(2, 3)
# print(x)
# print(x.size())
# print("transformed")
# z = x.view(1,2,3)
# print(z.size())
# print(z)
# print("transform back")
# w = z.view(2,3)
# print(w)
# print(w.size())





# import torch
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
#
# # 随机种子
# torch.manual_seed(0)
#
# # 定义边
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
#
# # 定义节点特征，每个节点特征维度是2
# x = torch.tensor([[-1,2], [0,4], [1,5]], dtype=torch.float)
#
#
# #创建一层GCN层，并把特征维度从2维降到1维
# conv = GCNConv(2, 1)
#
# # 前向传播
# x = conv(x, edge_index)
# print(x)


# tensor([[2.0028],
#         [3.1795],
#         [3.1302]], grad_fn=<AddBackward0>)


import  torch
import random

# x = torch.FloatTensor([[1,2,3],[0,1,2],[0,1,0]])
# index = torch.nonzero(x).tolist()
#index = index.tolist()
# print(index)
# print(index)
# print(index[:2])
# print("*"*10)
# print(index[2:])
#index = torch.cat([index[:,0].T,index[:1].T],dim=0)
#print(index[:,0])
# print(index[:,0].size())
# print(index[:,1].size())
# print(x)
# print("*"*10)
# x[index[:,0],index[:,1]] = -1
# print(x)
# print("*"*10)
# random.shuffle(index)
# print(index)
#print(x[index])
# print("*"*10)
# print(x[index])


# x = torch.FloatTensor([[1,2,1,0,0],[1,2,1,0,0]])
# mask = torch.eq(x,1)   # 非，一个数
# print(mask)
# x[mask] = -1
# print(x)

# x = t.zeros((3,2))
# m,n = x.size()
# print(m,n)
# from  torch import  nn
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print("*"*100)
# print(target)
# output = loss(input, target)
# print("*"*100)
# print(output)

# import  numpy as np
# import torch
# train_adj = torch.Tensor([[0,1,1],[0,0,1]])
#
# m = train_adj.shape[0]
# n = train_adj.shape[1]
#
# eval_coord = [(i,j) for i in range(m) for j in range(n)]
# train_edge_x , train_edge_y = train_adj.numpy().nonzero()
# eval_coord = set(eval_coord) - set(set(zip(train_edge_x, train_edge_y)))
# eval_coord = np.array(list(eval_coord))
# eval_coord = np.array(list(eval_coord))


# import torch as t
# from DataPreprocess import DataReader
# conv1 = t.nn.Conv2d(in_channels= 1 , out_channels= 2,kernel_size=(1,3),padding=(0,2))
# conv2 = t.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 3))
# X = DataReader.read_excel("../Data/Features/miRNAFeatureCompressed.xlsx")
# X = torch.relu(X)
# X = X.view(1,1,X.size(0),X.size(1))
# X = conv1(X)
# X = t.relu(X)
# X = conv2(X)
# X = t.relu(X)
# X = X.view(X.size(2), X.size(3))
# print(X.shape)
# print(X)

### roc_auc_score
import numpy as np
from sklearn.metrics import roc_auc_score
# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# ret = roc_auc_score(y_true, y_scores)


