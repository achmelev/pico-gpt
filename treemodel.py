import torch
import torch.nn as nn
from torch.nn import functional as F

from environment import log, get_int_config_value, workDir
from tokentree import TokenTree

class TokenTreeModel(nn.Module):

    def __init__(self):
        self.vocab_size = get_int_config_value('vocab_size')
        self.tree = TokenTree(workDir+'tokentree.bin','r')
        assert self.vocab_size==self.tree.vocab_size,'Wrong vocab size '+str(self.vocab_size)+"!="+str(self.tree.vocab_size)
        log.info('Initialized tree model vocab_size = '+str(self.vocab_size)+", tree size "+str(self.tree.size)+", tree depth = "+str(self.tree.depth))

    def __del__(self):
        self.tree.close()
    
    def create_ml_input(self, idx, inference=False):
        b, t = idx.size()
        ml_input = torch.zeros(b,t,self.tree.depth, self.vocab_size, dtype = torch.float)
        for b_idx in range(b):
            for t_idx in range(t):
                for tree_idx in range(self.tree.depth):
                    start_idx = t_idx-tree_idx
                    if (start_idx>=0):
                        sequence = idx[b_idx,start_idx:t_idx].tolist()
                        children = self.tree.getNodesChildren(sequence)
                        for token in children.keys():
                            node = children[token]
                            ml_input[b_idx, t_idx,tree_idx,token] = float(node.count)

        return ml_input


    
    def forward(self, idx, inference=False):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        ml_input = torch.zeroes(b,t,self.vocab_size,self.tree.depth)

        
        
        

