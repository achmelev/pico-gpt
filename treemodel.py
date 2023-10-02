import torch
import torch.nn as nn
from torch.nn import functional as F
from numpy import zeros, float32

from environment import log, get_int_config_value, workDir
from tokentree import TokenTree
from timers import create_timer, start, stop 

class TokenTreeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.vocab_size = get_int_config_value('vocab_size')
        self.block_size = get_int_config_value('block_size')
        self.tree = TokenTree(workDir+'tokentree.bin','r', cacheMaxSize=get_int_config_value('tree_cache_max_factor')*get_int_config_value('vocab_size'))
        assert self.vocab_size==self.tree.vocab_size,'Wrong vocab size '+str(self.vocab_size)+"!="+str(self.tree.vocab_size)
        log.info('Initialized tree model vocab_size = '+str(self.vocab_size)+", tree size "+str(self.tree.size)+", tree depth = "+str(self.tree.depth))
        self.linear = nn.Linear(self.tree.depth,1)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        self.np_array = None

    def __del__(self):
        self.tree.close()
    
    def create_ml_input(self, idx, inference=False):
        b, t = idx.size()
        if (self.np_array == None):
            if (inference):
                self.nparray = zeros((b,1,self.tree.depth, self.vocab_size),dtype=float32)
                t_start = t-1
            else:
                self.nparray = zeros((b,t,self.tree.depth, self.vocab_size),dtype=float32)
                t_start = 0

        zero_value = 0.0
        firstLevel = self.tree.getNodesChildren([])
        for b_idx in range(b):
            for t_idx in range(t_start,t):
                for tree_idx in range(self.tree.depth):
                    start_idx = t_idx-tree_idx
                    if (start_idx>=0):
                        sequence = idx[b_idx,start_idx:t_idx].tolist()
                        if (len(sequence)>0):
                            children = self.tree.getNodesChildren(sequence)
                        else:
                            children = firstLevel
                            
                        for token in children.keys():
                            node = children[token]
                            if (node.count == 0):
                                value = zero_value
                            else:
                                value = float32(node.count)
                            if (inference):
                                self.nparray[b_idx, 0,tree_idx,token] = value
                            else:
                                self.nparray[b_idx, t_idx,tree_idx,token] = value

        result =  torch.from_numpy(self.nparray)
        return result

    def forward(self, idx, inference=False):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        x = self.create_ml_input(idx, inference=inference)
        x = torch.transpose(x,2,3)
        #x = torch.log(x)
        #print(x)
        x = self.linear(x)
        x = torch.squeeze(x,3)
        return x
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

        
        
        

