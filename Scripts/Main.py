from DataPreprocess import *
import torch
from Model import MyModel
from ObjectiveFunction import  Myloss
from Evaluate import Evaluator
import numpy as np
import  time


# Hyperparameter
EPOCH = 5000
SEED = 123
KFold = 5
EVAL_INTER = 10
LR = 0.0001
USE_BIAS = False
KERNAL_SIZE1 = 512
KERNAL_SIZE2 = 256
TOLERANCE_EPOCH = 4000
STOP_THRESHOLD = 1e-5
ALPHA = 1
BETA = 1
GAMA = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

def main(miRNA_Disease_Association,disease_feature,disease_graph1,disease_graph2,disease_graph3,miRNA_feature,miRNA_graph1,miRNA_graph2,miRNA_graph3):

    adjProcess = adjTrainTestSplit(miRNA_Disease_Association)
    graph_train_kfold, graph_test_kfold = adjProcess.split_graph(KFold,SEED)

    auc_kfold = []
    aupr_kfold = []

    for i in range(KFold):
        print("Using {} th fold dataset.".format(i+1))
        graph_train = graph_train_kfold[i]
        graph_test = graph_test_kfold[i]

        m = graph_train.shape[0]
        n = graph_train.shape[1]
        eval_coord = [(i, j) for i in range(m) for j in range(n)]
        train_edge_x, train_edge_y = graph_train.nonzero()
        one_index = list(zip(train_edge_x, train_edge_y))
        zero_index = set(eval_coord) - set(set(zip(train_edge_x, train_edge_y)))
        zero_index = list(zero_index)

        adj_traget = torch.FloatTensor(graph_train)
        model = MyModel(disease_feature,disease_graph1,disease_graph2,disease_graph3,miRNA_feature,miRNA_graph1,miRNA_graph2,miRNA_graph3)
        model.cuda()
        obj = Myloss(adj_traget.cuda())
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True,weight_decay=GAMA)
        evaluator = Evaluator(graph_train, graph_test)
        obj_test = Myloss(torch.FloatTensor(graph_test).cuda())

        for j in range(EPOCH):
            model.train()
            optimizer.zero_grad()
            Y_hat, m_x0, m_x1, m_x2, m_x3, d_x0, d_x1, d_x2, d_x3 =  model()
            loss = obj.cal_loss(Y_hat, m_x0.cuda(), m_x1.cuda(), m_x2.cuda(), m_x3.cuda(), d_x0.cuda(), d_x1.cuda(), d_x2.cuda(), d_x3.cuda(),ALPHA,BETA,GAMA)
            loss = loss.cuda()
            # loss = obj.cal_loss(Y_hat,one_index,zero_index)
            #loss = obj.cal_loss(Y_hat,one_index,zero_index,m_x0,m_x1,m_x2,m_x3,d_x0, d_x1, d_x2,d_x3)
            loss.backward()

            optimizer.step()

            need_early_stop_check = j > TOLERANCE_EPOCH and abs((loss.item() - last_loss) / last_loss) < STOP_THRESHOLD
            if (j % EVAL_INTER == 0) or need_early_stop_check or j+1 >= EPOCH:
                t = time.time()
                model.eval()
                with torch.no_grad():
                    Y_hat, m_x0, m_x1, m_x2, m_x3, d_x0, d_x1, d_x2, d_x3 = model()
                    #test_loss = obj_test.cal_loss(Y_hat, m_x0, m_x1, m_x2, m_x3, d_x0, d_x1, d_x2, d_x3,ALPHA,BETA)
                   # Y_hat = torch.sigmoid(Y_hat)
                    eval_coord = [(i, j) for i in range(m) for j in range(n)]
                    test_edge_x, test_edge_y = graph_test.nonzero()
                    test_one_index = list(zip(test_edge_x, test_edge_y))
                    test_zero_index = set(eval_coord) - set(set(zip(test_edge_x, test_edge_y)))
                    test_zero_index = list(test_zero_index)
                    #test_loss = obj_test.cal_loss(Y_hat, test_one_index, test_zero_index)
                    auc_test, aupr_test = evaluator.eval(Y_hat.cpu())

                    print(
                        "Epoch:", '%04d' % (j + 1),
                        "train_loss=", "{:0>9.5f}".format(loss.item()),
                        "test_auc=", "{:.5f}".format(auc_test),
                        "test_aupr=", "{:.5f}".format(aupr_test),
                        "time=", "{:.2f}".format(time.time() - t))
                if need_early_stop_check or j+1 >= EPOCH:
                    auc_kfold.append(auc_test)
                    aupr_kfold.append(aupr_test)
                    if need_early_stop_check:
                        print("Early stopping...")
                    else:
                        print("Arrived at the last Epoch...")
                    break

            last_loss = loss.item()

    print("\nOptimization Finished!")
    mean_auc = sum(auc_kfold)/len(auc_kfold)
    mean_aupr = sum(aupr_kfold)/len(aupr_kfold)
    print("mean_auc:{0:.3f},mean_aupr:{1:.3f}".format(mean_auc,mean_aupr))

if __name__ == '__main__':
    # load data of miRNA
    # MSS = DataReader.read_npy("../Data/miRNA/MS_Sequence.npy")
    MSS = DataReader.read_npy("../Data/miRNA/MS_SequenceMean.npy")
    MSS = GraphPreprecess(MSS)

    MSG = DataReader.read_npy("../Data/miRNA/MS_GO.npy")
    MSG = GraphPreprecess(MSG)

    MSF = DataReader.read_npy("../Data/miRNA/MS_Function.npy")
    MSF = GraphPreprecess(MSF)

    M_F = DataReader.read_npy("../Data/Features/miRNAFeatureCompressed.npy")

    #load data of disease
    # DSP = DataReader.read_npy("../Data/Disease/DS_Phenotype.npy")
    DSP = DataReader.read_npy("../Data/Disease/DS_PhenotypeMean.npy")
    DSP = GraphPreprecess(DSP)

    DSSE = DataReader.read_npy("../Data/Disease/DS_Semantic.npy")
    DSSE = GraphPreprecess(DSSE)

    DSSY = DataReader.read_npy("../Data/Disease/DS_Symptom.npy")
    DSSY = GraphPreprecess(DSSY)

    D_F = DataReader.read_npy("../Data/Features/DiseaseFeatureCompressed.npy")


    #load data of miRNA-disease association
    M_D = DataReader.read_npy("../Data/MD.npy")
    t.manual_seed(1)
    M_F = t.eye(M_D.shape[0], M_D.shape[0])
    D_F = t.eye(M_D.shape[1], M_D.shape[1])
    main(M_D,D_F,DSSY,DSSE,DSP,M_F,MSS,MSF,MSG)
