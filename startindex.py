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
            self.data = memmap(workDir+'startindex.bin', dtype=uint32, mode='r')
            self.length = len(self.data)
        else:
            assert isfile(workDir+"train.bin"), "train data file found!"
            self.data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
            self.length = len(self.data)
    
    def getRandomPos(self):
        assert self.readonly,'is in writing mode'
        idx = randint(0, self.length-1)
        return int(self.data[idx])
    
    def generate(self, startToken):
        assert not self.readonly,'is in read mode'

        log.info('Generating startindex.')
        log.info('Scanning train data file...')
        result = []
        
        progress = Progress(self.length-1, 100)
        for idx  in range(self.length):
            if self.data[idx] == startToken:
                result.append(idx)
            progress.update(idx)
        log.info('Scanning done. Got '+str(len(result))+" positions")
        log.info("Writing to file...")
        arr =  array(result, dtype = uint32)
        f = open(workDir+"startindex.bin","wb")
        arr.tofile(f)
        f.close()
        log.info('Done!')
