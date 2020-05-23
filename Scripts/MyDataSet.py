import torch as t
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from Configure import Config


class Train_Test_Split(object):
    def __init__(self,path,split_rate = 0.5):
        self.data = pd.read_csv(path,sep='\t',index_col= 0)
        self.split_rate = split_rate

    def train_test_split(self):
        Train_Data = self.data.sample(frac=self.split_rate)
        rowlist = []
        for indexs in Train_Data.index:
            rowlist.append(indexs)
        Test_Data = self.data.drop(rowlist, axis=0)
        return t.FloatTensor(Train_Data.values),t.FloatTensor(Test_Data.values)


class MyDataSet(Dataset):
    def __init__(self,data):
        super(MyDataSet,self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index,:]

    def __len__(self):
        return len(self.data)


def main():
    config = Config()
    MySplit = Train_Test_Split("../Data/Features/miRNAFeature.txt", 0.7)
    Train_data, Test_Data = MySplit.train_test_split()
    train_dataset = MyDataSet(Train_data)
    test_dataset = MyDataSet(Test_Data)
    print(len(train_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    print(len(train_loader))
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    print(len(test_loader))
    for feature in test_loader:
        print(feature)

if __name__ == '__main__':
    main()