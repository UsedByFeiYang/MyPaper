from sklearn.metrics import roc_auc_score, precision_recall_curve, auc,roc_curve
import numpy as np



class Evaluator:
    def __init__(self, train_adj, test_adj):

        m = train_adj.shape[0]
        n = train_adj.shape[1]

        eval_coord = [(i,j) for i in range(m) for j in range(n)]
        train_edge_x , train_edge_y = train_adj.nonzero()
        eval_coord = set(eval_coord) - set(set(zip(train_edge_x, train_edge_y)))
        self.eval_coord = np.array(list(eval_coord))
        self.y_true = test_adj[self.eval_coord[:, 0], self.eval_coord[:, 1]]


    def eval(self, y_hat):
        y_score = y_hat[self.eval_coord[:, 0], self.eval_coord[:, 1]].numpy()
        auc_test = roc_auc_score(self.y_true, y_score)
        fpr, tpr, thresholds = roc_curve(self.y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(self.y_true, y_score)
        aupr_test = auc(recall, precision)
        return auc_test, aupr_test, fpr, tpr


