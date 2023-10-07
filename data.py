from environment import log, get_int_config_value, workDir, device
from os.path import isfile, isdir 
from numpy import memmap, uint16, int64
from torch import stack, from_numpy, randint

class DataLoader:

    def __init__(self):
        assert isfile(workDir+'train.bin'), 'Missing train data file'
        assert isfile(workDir+'val.bin'), 'Missing validation data file'
        self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
        self.val_data = memmap(workDir+'val.bin', dtype=uint16, mode='r')
        self.block_size = get_int_config_value("block_size")
        self.batch_size = get_int_config_value("batch_size")
    
    def batch(self, train = True):
        data = self.train_data if train else self.val_data
        ix = randint(len(data) - self.block_size, (self.batch_size,))
        samples = stack([from_numpy((data[i:i+self.block_size]).astype(int64)) for i in ix])
        # Die Targets sind die Tokens NACH eine Tokenfolge von 1 bis block size (max. context). 
        targets = stack([from_numpy((data[i+1:i+1+self.block_size]).astype(int64)) for i in ix])
        if device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            samples, targets = samples.pin_memory().to(device, non_blocking=True), targets.pin_memory().to(device, non_blocking=True)
        else:
            samples, targets = samples.to(device), targets.to(device)
        return samples, targets

    