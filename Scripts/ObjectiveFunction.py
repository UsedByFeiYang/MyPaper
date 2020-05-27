from torch.nn.functional import binary_cross_entropy_with_logits
import torch as t

class Myloss(object):

    def __init__(self,target):
        self.y_true = target

    def cal_loss(self, y_hat,m_x0,m_x1,m_x2,m_x3,d_x0, d_x1, d_x2,d_x3,alpha,beta):
        loss = binary_cross_entropy_with_logits(y_hat,self.y_true,reduction='mean')
        origin_loss = t.norm(m_x0 - m_x1) + t.norm(m_x0 - m_x1) + t.norm(m_x0 - m_x2) +t.norm(m_x0 - m_x2) + t.norm(d_x0 - d_x1) + t.norm(d_x0 - d_x2) + t.norm(d_x0 - d_x3)
        inter_loss = t.norm(m_x1 - m_x2) + t.norm(m_x1 - m_x3) + t.norm(m_x2 - m_x3) + t.norm(d_x1 - d_x2) + t.norm(d_x1 - d_x3) + t.norm(d_x2 - d_x3)
        return loss + alpha*origin_loss - beta * inter_loss