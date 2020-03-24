import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self,opt,tag):
        self.mean_every=opt.mean_every if opt.mean_every else 2000
        self.tag=tag
        self.num_nodes=opt.tgt_vocab_size
        self.att=[torch.zeros(self.num_nodes,self.num_nodes),0]
        self.loss_log=[]
        self.att_log=[]
        self.metrics_log=[]
        self.log={'tag':tag,'loss':self.loss_log,'att':self.att_log,
                  'metrics':self.metrics_log}
    def push_metrics(self,metrics):
        self.metrics_log.append(metrics)
    def push_loss(self,loss):
        self.loss_log.append(loss)
    def push_attentions(self,att):
        self.att[0]+=att.cpu().sum(axis=0)
        self.att[1]+=att.size(0)
        if self.att[1]>=self.mean_every:
            self.att_log.append(self.att[0]/self.att[1])
            self.att=[torch.zeros(self.num_nodes,self.num_nodes),0]
            
class Summary:
    def __init__(self,opt):
        self.exp_name=opt.exo_name if opt.exo_name else "base_exp"
        
        self.sw=SummaryWriter('experiments/'+self.exp_name)
    def add_log(self,log):
        tag=self.exp_name+log['tag']
        #print(log)
        #for i,loss in enumerate(log['loss']):
        #    self.sw.add_scalars(tag,{'loss':loss},i)
        for i,md in enumerate(log['metrics']):
            self.sw.add_scalars(tag,md,i)
        for i,att in enumerate(log['att']):
            self.sw.add_image(tag,att,i)
        
    def close(self):
        self.sw.close()