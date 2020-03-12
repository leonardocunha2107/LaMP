import torch

class TrimHandler:
    def __init__(self,num_nodes,initial_graph=None,crop_every=100,eps=0.5,cache_every=500):
        self.graph = initial_graph if initial_graph else torch.ones(num_nodes,num_nodes)
        self.crop_every=crop_every
        self.cache_every=cache_every
        self.num_nodes,self.n=num_nodes,num_nodes
        self.eps=eps
        
        self.cache=torch.zeros(num_nodes,num_nodes)
        self.t=0
        #self.log=[torch.zeros(num_nodes,num_nodes)]
    def push(self,attns):
        self.t+=1
        self.cache+=attns
        #self.log[-1]+=attns.data()  ##MWW
        
        """if self.t%self.cache_every==0:
            self.log[-1]/=self.cache_every ##maybe will create disappearing of attentions in graph?
            self.log.append(self.cache=torch.zeros(num_nodes,num_nodes))
        """
        if self.t%self.crop_every==0:
            self.cache/=self.cache.sum()
            idx=self.cache<self.eps/self.n  ##eps * 1/n
            self.graph[idx]=0
            self.n-=idx.sum().data()

            self.cache=torch.zeros(self.num_nodes,self.num_nodes)
    def get_mask(self,batch_size):
        with torch.no_grad():
            return (1-self.graph).unsqueeze(0).repeat(batch_size,1,1)
        
        
        
        
        
        