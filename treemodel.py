import torch
import torch.nn as nn
from torch.nn import functional as F

from environment import log, get_int_config_value, workDir
from tokentree import TokenTree

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

    def __del__(self):
        self.tree.close()
    
    def create_ml_input(self, idx, inference=False):
        b, t = idx.size()
        if (inference):
            ml_input = torch.zeros(b,1,self.tree.depth, self.vocab_size, dtype = torch.float)
            t_start = t-1
        else:
            ml_input = torch.zeros(b,t,self.tree.depth, self.vocab_size, dtype = torch.float)
            t_start = 0

        for b_idx in range(b):
            for t_idx in range(t_start,t):
                for tree_idx in range(self.tree.depth):
                    start_idx = t_idx-tree_idx
                    if (start_idx>=0):
                        sequence = idx[b_idx,start_idx:t_idx].tolist()
                        children = self.tree.getNodesChildren(sequence)
                        for token in children.keys():
                            node = children[token]
                            if (inference):
                                ml_input[b_idx, 0,tree_idx,token] = float(node.count)
                            else:
                                ml_input[b_idx, t_idx,tree_idx,token] = float(node.count)

        return ml_input

    def forward(self, idx, inference=False):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        ml_input = self.create_ml_input(idx, inference=inference)
        x = torch.transpose(ml_input,2,3)
        x = self.linear(x)
        x = torch.squeeze(x,3)
        return x

        
        
        

