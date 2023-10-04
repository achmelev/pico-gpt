from environment import log, get_int_config_value, workDir, device
from tokentree import TokenTree
from os.path import isfile
from numpy import memmap, uint16, array
import math

class TokenTreezer:

    def __init__(self):
        assert isfile(workDir+'train.bin'), 'Missing train data file'
        self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
    
    def int_list_from_numpy(self, narray):
        result = []
        for i in range(len(narray)):
            result.append(int(narray[i]))
        return result
    
    def createNumberFromTokenPath(self, tokenPath):
        assert len(tokenPath) >0, 'Empty token path not allowed'

        result = 0
        for pos in range(len(tokenPath)):
            digit = tokenPath[-pos-1]
            base = 1
            for _ in range(pos):
                base*=self.tree.vocab_size
            result+=(base*digit)
        return result

    def onTraverse(self, nodePath):
        
        level = len(nodePath)
        assert self.tree.depth >= level,'Wrong level '+len(level)+">"+str(self.tree.depth)
        if (level == self.tree.depth):
            self.lastLevelDictCount+=1
            if (self.lastLevelDictCount%100000 == 0):
                log.debug('Got '+str(self.lastLevelDictCount)+" Nodes")
            tokenPath = []
            for n in nodePath:
                tokenPath.append(n.token)
            self.lastLevelDict[self.createNumberFromTokenPath(tokenPath)] = n.index
    
    def createLastLevelDict(self):
        log.info('Preparing append, creating last level dict')
        self.lastLevelDictCount = 0
        self.lastLevelDict = {}
        self.tree.traverse(self)
        log.info('Done. Got '+str(self.lastLevelDictCount)+' last level nodes in the dict')
    
    def getLastLevelNode(self, tokenPath):
        return self.lastLevelDict[self.createNumberFromTokenPath(tokenPath)]

    def write(self, depth):
        assert depth>0, 'illegal depth '+str(depth)
        tree = TokenTree(workDir+"tokentree.bin",'w')
        self.tree = tree
        initialTreeDepth = tree.depth
        if (tree.depth == 0):
            tree.initFirstLevel(get_int_config_value('vocab_size'))
            log.info("Created and initialized a tree")
        else:
            log.info("Starting with existing tree of depth "+ str(tree.depth)+", and size = "+str(tree.size))

        if (depth <= initialTreeDepth):
            log.warning('Tree already filled up to depth '+ str(tree.depth)+ ', quitting.')
            tree.close()
            return
        elif (depth == initialTreeDepth+1):
            log.info('Appending one level...')
        else:
            log.error('Appending multiple levels not supported at the moment!')
            return

        #When the tree is new there ist no last level 
        if (depth > 1):
            self.createLastLevelDict()

        
        for idx in range(len(self.train_data)-depth+1):
            if (idx%1000 == 0):
                log.debug("Train data index is "+str(idx)+", tree size "+str(tree.size))
            sequence = self.int_list_from_numpy(self.train_data[idx:idx+depth])
            if (depth == 1):#We write the first level
                tree.insertOrUpdateToken(self.int_list_from_numpy(sequence))
            else:
                parentIdx = self.getLastLevelNode(sequence[:-1])
                tree.insertOrUpdateTokenFromParentIndex(parentIdx, sequence)

        log.info('Done. Got a tree with size = '+str(tree.size))
        tree.close()
    
