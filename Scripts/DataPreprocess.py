import pandas as pd
import torch as t
import  numpy as np
from sklearn.model_selection import  KFold


class DataReader(object):

    @staticmethod
    def read_excel(path):
        df = pd.read_excel(path)
        if "Unnamed: 0" in df.columns.tolist():
            del df['Unnamed: 0']
        return t.FloatTensor(df.values)

    @staticmethod
    def read_npy(path):
        df = np.load(path)
        return t.FloatTensor(df)

    @staticmethod
    def read_txt(path):
        df = pd.read_csv(path, sep='\t', index_col=0)
        if "Unnamed: 0" in df.columns.tolist():
            del df['Unnamed: 0']
        return t.FloatTensor(df.values)


class GraphPreprecess(object):
    def __init__(self,dataset):
        self.data = dict()
        self.data['weight'] = dataset
        self.data['edge'] = self.get_edge_index(dataset)

    def get_edge_index(self, matrix):
        edge_index = []
        x,y = matrix.numpy().nonzero()
        edge_index.append(x.tolist())
        edge_index.append(y.tolist())
        return t.LongTensor(edge_index)


class adjTrainTestSplit():

    def __init__(self,datatset):
        self.pairs  = datatset.nonzero().numpy()
        self.shape0 = datatset.shape[0]
        self.shape1 = datatset.shape[1]

    def split_graph(self,kfold, seed):
        if kfold is not None:
            prng = np.random.RandomState(seed)
            kf = KFold(n_splits=kfold, random_state=prng, shuffle=True)
            graph_train_kfold = []
            graph_test_kfold = []
            for train_indices, test_indices in kf.split(self.pairs):
                graph_train = np.zeros((self.shape0, self.shape1))
                graph_test = np.zeros((self.shape0, self.shape1))

                pair_x_train, pair_y_train = self.pairs[train_indices, 0], self.pairs[train_indices, 1]
                graph_train[pair_x_train, pair_y_train] = 1
                # graph_train[pair_y_train, pair_x_train] = 1

                pair_x_test, pair_y_test = self.pairs[test_indices, 0], self.pairs[test_indices, 1]
                graph_test[pair_x_test, pair_y_test] = 1


                graph_train_kfold.append(graph_train)
                graph_test_kfold.append(graph_test)

            return graph_train_kfold, graph_test_kfold

        else:
            graph_train = np.zeros((self.shape0, self.shape1))
            pair_x_train, pair_y_train = self.pairs[:, 0], self.pairs[:, 1]
            graph_train[pair_x_train, pair_y_train] = 1
            return graph_train

if __name__ == '__main__':
    # MSS = pd.read_excel("../Data/miRNA/MS_Function.xlsx")
    # MSS = t.FloatTensor(MSS.values)
    # ds = GraphPreprecess(MSS)
    # print(ds.data['edge'])
    # print(ds.data['edge'].size(1))

    MD = DataReader.read_excel("../Data/MD.xlsx")
    adj = adjTrainTestSplit(MD)
    train,test = adj.split_graph(5,3)
    print(type(train[0]))
    print(train[0])
