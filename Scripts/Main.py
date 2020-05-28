from DataPreprocess import *
import torch
from Model import MyModel
from ObjectiveFunction import  Myloss
from Evaluate import Evaluator
import numpy as np
import  time


# Hyperparameter
EPOCH = 2000
SEED = 123
KFold = 5
EVAL_INTER = 50
LR = 0.01
USE_BIAS = False
KERNAL_SIZE1 = 512
KERNAL_SIZE2 = 256
TOLERANCE_EPOCH = 1000
STOP_THRESHOLD = 1e-5
ALPHA = 0
BETA = 0
GAMA = 1e-2
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


        adj_traget = torch.FloatTensor(graph_train)
        model = MyModel(disease_feature,disease_graph1,disease_graph2,disease_graph3,miRNA_feature,miRNA_graph1,miRNA_graph2,miRNA_graph3)
        obj = Myloss(adj_traget)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True,weight_decay=GAMA)
        evaluator = Evaluator(graph_train, graph_test)
        obj_test = Myloss(torch.FloatTensor(graph_test))

        for j in range(EPOCH):
            model.train()
            optimizer.zero_grad()
            Y_hat, m_x0, m_x1, m_x2, m_x3, d_x0, d_x1, d_x2, d_x3 =  model()
            loss = obj.cal_loss(Y_hat, m_x0, m_x1, m_x2, m_x3, d_x0, d_x1, d_x2, d_x3,ALPHA,BETA)
            loss.backward()

            optimizer.step()

            need_early_stop_check = j > TOLERANCE_EPOCH and abs((loss.item() - last_loss) / last_loss) < STOP_THRESHOLD
            if (j % EVAL_INTER == 0) or need_early_stop_check or j+1 >= EPOCH:
                t = time.time()
                model.eval()
                with torch.no_grad():
                    Y_hat, m_x0, m_x1, m_x2, m_x3, d_x0, d_x1, d_x2, d_x3 = model()
                    test_loss = obj_test.cal_loss(Y_hat, m_x0, m_x1, m_x2, m_x3, d_x0, d_x1, d_x2, d_x3,ALPHA,BETA)
                    Y_hat = torch.sigmoid(Y_hat)
                    auc_test, aupr_test = evaluator.eval(Y_hat)

                    print(
                        "Epoch:", '%04d' % (j + 1),
                        "train_loss=", "{:0>9.5f}".format(loss.item()),
                        "test_loss=", "{:0>9.5f}".format(test_loss.item()),
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
    MSS = DataReader.read_excel("../Data/miRNA/MS_Sequence.xlsx")
    MSS = GraphPreprecess(MSS)

    MSG = DataReader.read_excel("../Data/miRNA/MS_GO.xlsx")
    MSG = GraphPreprecess(MSG)

    MSF = DataReader.read_excel("../Data/miRNA/MS_Function.xlsx")
    MSF = GraphPreprecess(MSF)

    M_F = DataReader.read_excel("../Data/Features/miRNAFeatureCompressed.xlsx")

    #load data of disease
    DSP = DataReader.read_excel("../Data/Disease/DS_Phenotype.xlsx")
    DSP = GraphPreprecess(DSP)

    DSSE = DataReader.read_excel("../Data/Disease/DS_Semantic.xlsx")
    DSSE = GraphPreprecess(DSSE)

    DSSY = DataReader.read_excel("../Data/Disease/DS_Symptom.xlsx")
    DSSY = GraphPreprecess(DSSY)

    D_F = DataReader.read_excel("../Data/Features/DiseaseFeatureCompressed.xlsx")


    #load data of miRNA-disease association
    M_D = DataReader.read_excel("../Data/MD.xlsx")

    main(M_D,D_F,DSP,DSSE,DSSY,M_F,MSS,MSG,MSF)
