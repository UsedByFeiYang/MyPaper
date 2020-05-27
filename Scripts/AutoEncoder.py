import time
import numpy as np
import pandas as pd
import torch as  t
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader


class Config(object):
    def __init__(self,datapath,compressedfeaturepath ,random_seed = 1,learning_rate = 0.001,num_epochs = 50,batch_size = 32,num_features = 784,num_hidden_1 = 500,num_latent = 15):
        self.path = datapath
        self.compressedfeaturepath = compressedfeaturepath
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        # Architecture
        self.num_features = num_features
        self.num_hidden_1 = num_hidden_1
        self.num_latent = num_latent

class MyDataSet(Dataset):
    def __init__(self,data):
        super(MyDataSet,self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index,:]

    def __len__(self):
        return len(self.data)

class Train_Test_Split(object):

    def __init__(self,dataset,split_rate = 0.5):
        self.dataset = dataset
        self.split_rate = split_rate

    def train_test_split(self):
        Train_Data = self.dataset.sample(frac=self.split_rate)
        rowlist = []
        for indexs in Train_Data.index:
            rowlist.append(indexs)
        Test_Data = self.dataset.drop(rowlist, axis=0)
        return t.FloatTensor(Train_Data.values),t.FloatTensor(Test_Data.values)

class VariationalAutoencoder(t.nn.Module):
    def __init__(self,num_features,num_hidden_1,num_latent,device):
        super(VariationalAutoencoder,self).__init__()

        self.device = device
        #Encode
        self.hidden_1 = t.nn.Linear(num_features, num_hidden_1)
        self.z_mean = t.nn.Linear(num_hidden_1, num_latent)
        self.z_log_var = t.nn.Linear(num_hidden_1, num_latent)

        #Decode
        self.linear_3 = t.nn.Linear(num_latent, num_hidden_1)
        self.linear_4 = t.nn.Linear(num_hidden_1, num_features)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = t.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * t.exp(z_log_var / 2.)
        return z

    def encoder(self, features):
        x = self.hidden_1(features)
        x = F.leaky_relu(x, negative_slope=0.0001)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded

    def decoder(self, encoded):
        x = self.linear_3(encoded)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.linear_4(x)
        decoded = t.sigmoid(x)
        return decoded

    def forward(self, features):
        z_mean, z_log_var, encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded


def Load_Data(dataset,split_rate = 0.7,batch = 32):
    MySplit = Train_Test_Split(dataset, split_rate)
    Train_data, Test_Data = MySplit.train_test_split()
    train_dataset = MyDataSet(Train_data)
    test_dataset = MyDataSet(Test_Data)
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size= batch, shuffle=False)
    return train_loader,test_loader

def Train(model, optimizer, train_loader,num_epochs,device):
    start_time = time.time()
    for epoch in range(num_epochs):
        print("**** epoch:{0} ****".format(epoch))
        for batch_idx,features in enumerate(train_loader):

            # don't need labels, only the images (features)
            features = features.to(device)

            ### FORWARD AND BACK PROP
            z_mean, z_log_var, encoded, decoded = model(features)

            # cost = reconstruction loss + Kullback-Leibler divergence
            kl_divergence = (0.5 * (z_mean ** 2 +
                                    t.exp(z_log_var) - z_log_var - 1)).sum()
            pixelwise_bce = F.binary_cross_entropy(decoded, features, reduction='sum')
            cost = kl_divergence + pixelwise_bce

            optimizer.zero_grad()
            cost.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()
        print("cost:{0:.4f}".format(cost))
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
        print()

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

def Generate_compreseed_feature(model,dataset,num_latent,num_sample,device):
    X_compr = np.ones((num_sample, num_latent))

    start_idx = 0
    for idx, features in enumerate(dataset):
        features = features.to(device)
        *_, decoded = model.encoder(features)
        X_compr[start_idx:start_idx + len(features)] = decoded.to(t.device('cpu')).detach().numpy()
        start_idx += len(features)
    return X_compr


def run(config):

    dataset = pd.read_csv(config.path, sep='\t', index_col=0)
    train_loader, test_loader = Load_Data(dataset,0.7,config.batch_size)
    model = VariationalAutoencoder(config.num_features, config.num_hidden_1, config.num_latent,config.device)
    model = model.to(config.device)
    optimizer = t.optim.Adam(model.parameters(), lr=config.learning_rate)
    Train(model,optimizer,train_loader,config.num_epochs,config.device)

    #generate compressed feature
    dataset = t.FloatTensor(dataset.values)
    num_sample = len(dataset)
    dataset = MyDataSet(dataset)
    dataset = DataLoader(dataset=dataset, batch_size= config.batch_size , shuffle=False)
    Feature = Generate_compreseed_feature(model,dataset,config.num_latent,num_sample,config.device)
    Feature = pd.DataFrame(Feature)
    Feature.to_excel(config.compressedfeaturepath)


def main():
    Mirpath = "../Data/Features/miRNAFeature.txt"
    MirFeaturePath = "../Data/Features/miRNAFeatureCompressed.xlsx"
    configMir = Config(Mirpath,MirFeaturePath,random_seed=1, learning_rate=0.001, num_epochs=2000, batch_size=32, num_features=11198,
                    num_hidden_1=1000, num_latent=128)
    run(configMir)

    Dispath = "../Data/Features/DiseaseFeature.txt"
    DisFeaturePath = "../Data/Features/DiseaseFeatureCompressed.xlsx"
    configDis = Config(Dispath,DisFeaturePath,random_seed=1, learning_rate=0.001, num_epochs=2000, batch_size=32, num_features=11198,
                    num_hidden_1=1000, num_latent=128)
    run(configDis)



if __name__ == '__main__':
    main()
