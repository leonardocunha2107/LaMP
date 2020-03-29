import torch

class TrimHandler:
    def __init__(self,num_nodes,initial_graph=None,crop_every=100,eps=0.5):
        self.graph = initial_graph if initial_graph else torch.ones(num_nodes,num_nodes)
        self.crop_every=crop_every
        self.num_nodes,self.n=num_nodes,num_nodes**2
        self.eps=eps
        
        self.cache=torch.zeros(num_nodes,num_nodes)
        self.t=0
    def push(self,attns):
        self.t+=attns.shape[0]
        with torch.no_grad():
            self.cache+=attns.sum(axis=0).cpu()
        #self.log[-1]+=attns.data()  ##MWW
        
        """if self.t%self.cache_every==0:
            self.log[-1]/=self.cache_every ##maybe will create disappearing of attentions in graph?
            self.log.append(self.cache=torch.zeros(num_nodes,num_nodes))
        """
        if self.t>=self.crop_every:
            self.cache/=self.cache.sum()
            self.t=0
            print("Cache\n",self.cache)
            idx=self.cache<(self.eps/(self.n**2))  ##eps * 1/n
            self.graph[idx]=0
            self.n=self.num_nodes**2-int(idx.sum())
            print("Graph Updated\n",self.graph)
            print("num_connections",self.n)


            self.cache=torch.zeros(self.num_nodes,self.num_nodes)
    def get_mask(self,batch_size):
        with torch.no_grad():
            return (1-self.graph).type(torch.bool).unsqueeze(0).cuda().repeat(batch_size,1,1)
        
        
        
        
        
        