from torch.nn.functional import mse_loss
#from torch.nn.functional import binary_cross_entropy_with_logits
import torch as t

class Myloss(object):

    def __init__(self,target):
        self.y_true = target

    def cal_loss(self, y_hat,m_x0,m_x1,m_x2,m_x3,d_x0, d_x1, d_x2,d_x3,alpha,beta,gama):
        loss = mse_loss(y_hat,self.y_true,reduction = 'sum')
        # origin_loss = t.norm(m_x0 - m_x1) + t.norm(m_x0 - m_x1) + t.norm(m_x0 - m_x2) +t.norm(m_x0 - m_x2) + t.norm(d_x0 - d_x1) + t.norm(d_x0 - d_x2) + t.norm(d_x0 - d_x3)
        # inter_loss = t.norm(m_x1 - m_x2) + t.norm(m_x1 - m_x3) + t.norm(m_x2 - m_x3) + t.norm(d_x1 - d_x2) + t.norm(d_x1 - d_x3) + t.norm(d_x2 - d_x3)
        # #regular_loss = t.norm(m_x0) + t.norm(d_x0)
        return loss # + alpha*origin_loss - beta * inter_loss #+ gama * regular_loss
        # #return inter_loss
        # return  regular_loss


# from torch import  nn
#
# class Myloss(object):
#     def __init__(self,target):
#         self.y_true = target
#
#     def cal_loss(self, input, one_index, zero_index,alpha = 0):
#         loss = nn.MSELoss(reduction='none')
#         loss_sum = loss(input, self.y_true)
#         return (1 - alpha) * loss_sum[one_index].sum() + alpha * loss_sum[zero_index].sum()


# from torch import  nn
# import torch as t
# class Myloss(object):
#     def __init__(self,target):
#         self.y_true = target
#
#     def cal_loss(self, input, one_index, zero_index,m_x0,m_x1,m_x2,m_x3,d_x0, d_x1, d_x2,d_x3,alpha = 0.5,beta = 0.2,gama = 0.1):
#         loss = nn.MSELoss(reduction='none')
#         loss_sum = loss(input, self.y_true)
#         #origin_loss = t.norm(m_x0 - m_x1) + t.norm(m_x0 - m_x1) + t.norm(m_x0 - m_x2) + t.norm(m_x0 - m_x2) + t.norm(d_x0 - d_x1) + t.norm(d_x0 - d_x2) + t.norm(d_x0 - d_x3)
#         #inter_loss = t.norm(m_x1 - m_x2) + t.norm(m_x1 - m_x3) + t.norm(m_x2 - m_x3) + t.norm(d_x1 - d_x2) + t.norm(d_x1 - d_x3) + t.norm(d_x2 - d_x3)
#         #regular_loss = t.norm(m_x0) + t.norm(d_x0)
#         #return (1 - alpha) * loss_sum[one_index].sum() + alpha * loss_sum[zero_index].sum() + beta * inter_loss + gama * origin_loss + origin_loss
#         return (1 - alpha) * loss_sum[one_index].sum() + alpha * loss_sum[zero_index].sum() + beta