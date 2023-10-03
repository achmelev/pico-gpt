import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from numpy import zeros, float32, empty

from environment import log, get_int_config_value, workDir
from tokentree import TokenTree
from timers import create_timer, start, stop 
from functools import lru_cache

tree = None
zero_value = 0.0

@lru_cache(maxsize=get_int_config_value('treemodel_cache_size'))
def getNextTokenCounts(tokenPath):
    vocab_size = get_int_config_value('vocab_size')
    result = empty((vocab_size),dtype=float32)
    children =  tree.getNodesChildren(tokenPath)
        
    for token in range(vocab_size):
        if (token in children.keys()):
            node = children[token]
            if (node.count == 0):
                value = zero_value
            else:
                value = float32(node.count)
        else:
            value = zero_value 
        result[token] = value

    return result
        


class TokenTreeModel(nn.Module):

    def __init__(self, initZeroValue = 0.0):
        super().__init__()
        global tree, zero_value
        zero_value = initZeroValue
        assert tree == None,'This model is a singleton!'
        self.vocab_size = get_int_config_value('vocab_size')
        self.block_size = get_int_config_value('block_size')
        tree = TokenTree(workDir+'tokentree.bin','r')
        assert self.vocab_size==tree.vocab_size,'Wrong vocab size '+str(self.vocab_size)+"!="+str(tree.vocab_size)
        log.info('Initialized tree model vocab_size = '+str(self.vocab_size)+", tree size "+str(tree.size)+", tree depth = "+str(tree.depth))
        self.linear = nn.Linear(tree.depth,1)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        self.np_array = None
        self.tree = tree

    def __del__(self):
        global tree
        tree.close()
        tree = None
        getNextTokenCounts.cache_clear()
    
    def create_ml_input(self, idx, inference=False):
        global tree, zero_value
        b, t = idx.size()
        if (self.np_array == None):
            if (inference):
                self.nparray = empty((b,1,tree.depth, self.vocab_size),dtype=float32)
                t_start = t-1
            else:
                self.nparray = empty((b,t,tree.depth, self.vocab_size),dtype=float32)
                t_start = 0

        zeros = empty((self.vocab_size),dtype=float32)
        for token in range(self.vocab_size):
            zeros[token] = zero_value
        for b_idx in range(b):
            for t_idx in range(t_start,t):
                for tree_idx in range(tree.depth):
                    start_idx = t_idx-tree_idx
                    if (start_idx>=0):
                        sequence = idx[b_idx,start_idx:t_idx].tolist()
                        token_counts = getNextTokenCounts(tuple(sequence))
                    else:
                        token_counts = zeros

                    if  inference:
                        self.nparray[b_idx, 0,tree_idx] = token_counts
                    else:
                        self.nparray[b_idx, t_idx,tree_idx] = token_counts
                    


        result =  torch.from_numpy(self.nparray)
        return result

    def forward(self, idx, inference=False):
        global zero_value
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        x = self.create_ml_input(idx, inference=inference)
        x = torch.transpose(x,2,3)
        x = torch.log(x)-math.log(zero_value)
        #print(x)
        x = self.linear(x)
        x = torch.squeeze(x,3)
        return x
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

        
        
        

