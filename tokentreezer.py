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
        tree = TokenTree(workDir+"tokentree.bin",'w', cacheMaxSize=get_int_config_value('tree_cache_max_factor')*get_int_config_value('vocab_size'))
        initialTreeDepth = tree.depth
        if (tree.depth == 0):
            tree.initFirstLevel(get_int_config_value('vocab_size'))
            log.info("Created and initialized a tree")
        else:
            log.info("Starting with existing tree of depth "+ str(tree.depth)+", and size = "+str(tree.size))

        if (depth <= initialTreeDepth):
            log.warn('Tree already filled up to depth '+ str(tree.depth)+ ', quitting.')
            tree.close()
            return
        else:
            startDepth = initialTreeDepth+1
            stopDepth = depth
        
        for idx in range(len(self.train_data)):
            if (idx%1000 == 0):
                log.debug("Train data index is "+str(idx)+", tree size "+str(tree.size))
            maxl = min(stopDepth, len(self.train_data)-idx)
            sequence = self.train_data[idx:idx+maxl]
            for idx2 in range(startDepth,len(sequence)+1):
                subsequence = sequence[0:idx2]
                tree.insertOrUpdateToken(self.int_list_from_numpy(subsequence))
        
        log.info('Done. Got a tree with size = '+str(tree.size))
        tree.close()
