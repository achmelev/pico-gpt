from environment import log, workDir
from tokentree import TokenTree
from os.path import isfile

class TokenTreeReader:

    def __init__(self):
        self.tree = TokenTree(workDir+"tokentree.bin",'r')
        self.count = 0
        self.stats = []
    
    def onTraverse(self, nodePath):
        self.count+=1
        level = len(nodePath)
        assert self.tree.depth >= level,'Wrong level '+len(level)+">"+str(self.tree.depth)
        assert len(self.stats) >= level-1,'Wrong length stats '+len(self.stats)+"<"+str(level-1)
        if (len(self.stats) == level-1):
            self.stats.append(0)
        self.stats[level-1]+=1
        if (self.count%100000 == 0):
            log.debug('Got '+str(self.count)+" Nodes")

    def print_stats(self):
        log.info('Reading tree stats. Tree size is '+str(self.tree.size)+", tree depth is "+str(self.tree.depth))
        self.statsCounter = 0
        self.tree.traverse(self)
        log.info('Done, read '+str(self.count)+" Nodes")
        log.info('#######RESULT#######')
        for level in range(self.tree.depth):
            log.info('Level '+str(level+1)+' - '+str(self.stats[level])+' Nodes')

        
