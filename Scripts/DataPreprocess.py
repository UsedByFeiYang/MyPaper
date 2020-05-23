import pandas as pd
import torch as t
import random
import  numpy as np



class DataReader(object):

    @staticmethod
    def read_excel(path):
        df = pd.read_excel(path)
        return t.FloatTensor(df.values)

    @staticmethod
    def read_txt(path):
        df = pd.read_csv(path, sep='\t', index_col=0)
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
    @staticmethod
    def prepare_data(path,splitRate):
        dataset = dict()
        dataset['adj'] = DataReader.read_excel(path)


        m,n = dataset['adj'].size()
        trainLen = int(m * splitRate)

        trainWeight = t.zeros((m,n))
        testWeight = t.zeros((m,n))

        one_index = t.nonzero(dataset['adj']).tolist()

        random.shuffle(one_index)

        train_one_index = one_index[:trainLen]
        test_one_index  = one_index[trainLen:]

        train_one_index_tensor = t.LongTensor(train_one_index)
        test_one_index_tensor = t.LongTensor(test_one_index)

        trainWeight[train_one_index_tensor[:,0],train_one_index_tensor[:,1]] = 1
        testWeight[test_one_index_tensor[:,0],test_one_index_tensor[:,0]]  = 1

        dataset['train'] = trainWeight
        dataset['test'] = testWeight
        return dataset

def split_graph(kfold, pairs, num_node, seed):
    if kfold is not None:
        prng = np.random.RandomState(seed)
        kf = KFold(n_splits=kfold, random_state=prng, shuffle=True)

        graph_train_kfold = []
        graph_test_kfold = []
        for train_indices, test_indices in kf.split(pairs):
            graph_train = np.zeros((num_node, num_node))
            graph_test = np.zeros((num_node, num_node))

            pair_x_train, pair_y_train = pairs[train_indices, 0], pairs[train_indices, 1]
            graph_train[pair_x_train, pair_y_train] = 1
            graph_train[pair_y_train, pair_x_train] = 1

            pair_x_test, pair_y_test = pairs[test_indices, 0], pairs[test_indices, 1]
            graph_test[pair_x_test, pair_y_test] = 1
            graph_test[pair_y_test, pair_x_test] = 1

            graph_train_kfold.append(graph_train)
            graph_test_kfold.append(graph_test)

        return graph_train_kfold, graph_test_kfold

    else:
        graph_train = np.zeros((num_node, num_node))
        pair_x_train, pair_y_train = pairs[:, 0], pairs[:, 1]
        graph_train[pair_x_train, pair_y_train] = 1
        graph_train[pair_y_train, pair_x_train] = 1

        return graph_train

if __name__ == '__main__':
    MSS = pd.read_excel("../Data/miRNA/MS_Function.xlsx")
    MSS = t.FloatTensor(MSS.values)
    ds = GraphPreprecess(MSS)
    # ds.Get_Data(ds)
    print(ds.data['edge'])
    print(ds.data['edge'].size(1))