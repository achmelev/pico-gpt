import unittest
from environment import initEnv, get_int_config_value
from shutil import rmtree
from torch.nn import functional as F
from tokentree_test import RandomTokenTree
import numpy as np
from timers import create_timer, start, stop, get_time_sum_fmt, get_time_avg_fmt

class TokenTreeModelPFTest (unittest.TestCase):

     def setUp(self) -> None:
        initEnv('unittest')
        from environment import log, workDir
        from tokenizer import Tokenizer
        from tokentreezer import TokenTreezer
        #preparing data
        log.debug('TEST VOCAB')
        tokenizer1 = Tokenizer()
        input_file = __file__[:__file__.rfind('/')]+'/testdaten/erlkoenig_input.txt'
        tokenizer1.generate_vocab(input_file)
        tokenizer1.tokenize(input_file)

        #Writing tree
        tokentreezer = TokenTreezer()
        tokentreezer.write(3)
    
     def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    
     def test_performance(self):
        from environment import log
        log.debug('TEST ML INPUT')
        from data import DataLoader
        from treemodel import TokenTreeModel, getNextTokenCounts
        loader = DataLoader()
        model = TokenTreeModel()

        create_timer('ml_input')

        for i in range(10):
            train_batch = loader.batch()
            start('ml_input')
            ml_input = model.create_ml_input(train_batch[0])
            stop('ml_input')
        
        log.info('ML Input time '+get_time_sum_fmt('ml_input'))
        log.info('Cache info = '+str(getNextTokenCounts.cache_info()))

if __name__ == '__main__':
    unittest.main()      