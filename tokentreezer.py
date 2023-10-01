from environment import log, get_int_config_value, workDir, device
from tokentree import TokenTree
from os.path import isfile
from numpy import memmap, uint16, int64

class TokenTreezer:

    def __init__(self):
        assert isfile(workDir+'train.bin'), 'Missing train data file'
        self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
    
    def int_list_from_numpy(self, narray):
        result = []
        for i in range(len(narray)):
            result.append(int(narray[i]))
        return result

    def write(self, depth):
        assert depth>0, 'illegal depth '+str(depth)
        tree = TokenTree(workDir+"tokentree.bin",'w')
        if (tree.depth == 0):
            tree.initFirstLevel(get_int_config_value('vocab_size'))
            log.info("Created and initialized a tree")
        else:
            log.info("Starting with existing tree of depth "+ str(tree.depth)+", and size = "+str(tree.size))
        if (depth <= tree.depth):
            log.warn('Tree already filled up to depth '+ str(tree.depth)+ ', quitting.')
        else:
            startDepth = tree.depth+1
            stopDepth = depth
        
        for idx in range(len(self.train_data)):
            maxl = min(stopDepth, len(self.train_data)-idx)
            sequence = self.train_data[idx:idx+maxl]
            for idx2, _ in enumerate(sequence):
                subsequence = sequence[0:idx2+1]
                if (len(subsequence) >= startDepth):
                    tree.insertOrUpdateToken(self.int_list_from_numpy(subsequence))
                    if (tree.size%1000 == 0):
                        log.debug('Got '+str(tree.size)+", nodes, current index in train data is = "+str(idx))
        
        log.info('Done. Got a tree with size = '+str(tree.size))
        tree.close()
