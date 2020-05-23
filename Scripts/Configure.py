import  torch as t
class Config(object):
    def __init__(self,random_seed = 1,learning_rate = 0.001,num_epochs = 50,batch_size = 32,num_features = 784,num_hidden_1 = 500,num_latent = 15):
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        # Architecture
        self.num_features = num_features
        self.num_hidden_1 = num_hidden_1
        self.num_latent = num_latent