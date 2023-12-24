from environment import log, workDir, get_int_config_value

from os.path import isfile, isdir 
from numpy import memmap, uint32,uint16,array
from random import randint
from progress import Progress

class StartIndex:

    def __init__(self, readonly = True):
        self.readonly = readonly
        if (self.readonly):
            assert isfile(workDir+"startindex.bin"), "no startindex file found!"
            assert isfile(workDir+"startindex_val.bin"), "no startindex val file found!"
            self.data = memmap(workDir+'startindex.bin', dtype=uint32, mode='r')
            self.length = len(self.data)
            self.data_val = memmap(workDir+'startindex_val.bin', dtype=uint32, mode='r')
            self.length_val = len(self.data_val)
        else:
            assert isfile(workDir+"train.bin"), "train data file found!"
            assert isfile(workDir+"val.bin"), "val data file found!"
            self.data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
            self.length = len(self.data)
            self.data_val = memmap(workDir+'val.bin', dtype=uint16, mode='r')
            self.length_val = len(self.data_val)
            
            
    def getRandomPos(self):
        assert self.readonly,'is in writing mode'
        idx = randint(0, self.length-1)
        return int(self.data[idx])
    
    def getValRandomPos(self):
        assert self.readonly,'is in writing mode'
        idx = randint(0, self.length_val-1)
        return int(self.data_val[idx])
    
    def do_generate(self, startToken, train):
        assert not self.readonly,'is in read mode'

        
        if train:
            log.info('Scanning train data file...')
        else:
            log.info('Scanning validation data file...')
        result = []
        
        if train:
            length = self.length
            data = self.data
            out = workDir+"startindex.bin"
        else:
            length = self.length_val
            data = self.data_val
            out = workDir+"startindex_val.bin"

        progress = Progress(length-1, 100)
        for idx  in range(length):
            if data[idx] == startToken:
                result.append(idx)
            progress.update(idx)
        log.info('Scanning done. Got '+str(len(result))+" positions")
        log.info("Writing to file...")
        arr =  array(result, dtype = uint32)
        f = open(out,"wb")
        arr.tofile(f)
        f.close()
        log.info('Done!')
    
    def generate(self, startToken):
        log.info('Generating startindex.')
        self.do_generate(startToken=startToken, train=True)
        self.do_generate(startToken=startToken,train=False)

