import torch

class Logger:
    def __init__(self,opt):
        self.mean_every=opt.mean_every if opt.mean_every else 2000
        self.num_nodes=opt.n_tgt_vocab 
        self.att=[torch.zeros(self.num_nodes,self.num_nodes),0]
        self.loss_log=[]
        self.att_log=[]
        self.metrics_log=[]
        self.log={'loss':self.loss_log,'att':self.att_log,'metrics':self.metrics}
        
    def push_loss(self,loss):
        self.loss_log.append(loss.data())
    def push_attentions(self,att):
        self.att[0]+=att.sum(axis=0)
        self.att[1]+=att.size(0)
        if self.att[1]>=self.mean_every:
            self.att_log.append(self.att[0])
            self.att=[torch.zeros(self.num_nodes,self.num_nodes),0]
            
        
        