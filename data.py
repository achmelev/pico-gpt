from environment import log, get_int_config_value,get_bool_config_value, workDir, device
from os.path import isfile, isdir 
from numpy import memmap, uint16, int64
from torch import stack, from_numpy, randint,tensor
import torch
from startindex import StartIndex, hasStartIndex

class DataLoader:

    def __init__(self, useStartIndex = None, validationOff = None):
        #Without validation set
        if (validationOff == None):
            self.validationOff = get_bool_config_value('validation_off')
        else:
            self.validationOff = validationOff
        assert isfile(workDir+'train.bin'), 'Missing train data file'
        if not self.validationOff:
            assert isfile(workDir+'val.bin'), 'Missing validation data file'
        self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
        if not self.validationOff:
            self.val_data = memmap(workDir+'val.bin', dtype=uint16, mode='r')
        self.block_size = get_int_config_value("block_size")
        self.batch_size = get_int_config_value("batch_size")
        self.useStartIndex = useStartIndex
        if (self.useStartIndex == None):
            self.useStartIndex = get_bool_config_value("use_start_index")
        if (self.useStartIndex):
            assert hasStartIndex(),"no start index found"
            log.info("Getting samples start positions from index")
            self.startIndex = StartIndex(readonly=True, rightPadding=self.block_size)

    def getStartIndexes(self, train):
        if self.useStartIndex:
            result = self.startIndex.getRandomPos(count=self.batch_size, train=train)
            return tensor(data=result, dtype = torch.int64)
        else:
            data = self.train_data if train else self.val_data
            return randint(len(data) - self.block_size, (self.batch_size,))
    
    def batch(self, train = True):
        if not train:
            assert not self.validationOff,'validation file disabled'
        data = self.train_data if train else self.val_data
        ix = self.getStartIndexes(train)
        samples = stack([from_numpy((data[i:i+self.block_size]).astype(int64)) for i in ix])
        # Die Targets sind die Tokens NACH eine Tokenfolge von 1 bis block size (max. context). 
        targets = stack([from_numpy((data[i+1:i+1+self.block_size]).astype(int64)) for i in ix])
        if device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            samples, targets = samples.pin_memory().to(device, non_blocking=True), targets.pin_memory().to(device, non_blocking=True)
        else:
            samples, targets = samples.to(device), targets.to(device)
        return samples, targets

    