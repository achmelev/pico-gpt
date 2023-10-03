import unittest
from environment import initEnv, get_int_config_value, get_float_config_value
from shutil import rmtree
from torch.nn import functional as F
from tokentree_test import RandomTokenTree
import numpy as np

class TokenTreeModelTest (unittest.TestCase):

    def setUp(self) -> None:
        initEnv('unittest')
        from environment import log, workDir
        from tokenizer import Tokenizer
        from tokentreezer import TokenTreezer
        log.debug('Preparing data')
        #Preparing train file
        self.rtree = RandomTokenTree()
        self.rtree.createRandomTestData(5,500, vocab_size=get_int_config_value('vocab_size'))
        np_tokens = np.array(self.rtree.tokens, dtype=np.uint16)
        trf = open(workDir+"train.bin",'wb')
        np_tokens.tofile(trf)
        trf.close()
        trf = open(workDir+"val.bin",'wb')
        np_tokens.tofile(trf)
        trf.close()

        #Writing tree
        tokentreezer = TokenTreezer()
        tokentreezer.write(5)

    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    
    def test_ml_input(self):
        from environment import log
        log.debug('TEST ML INPUT')
        from data import DataLoader
        from treemodel import TokenTreeModel
        loader = DataLoader()
        model = TokenTreeModel(get_float_config_value('treemodel_zero_value'))
        train_batch = loader.batch()

        block_size = get_int_config_value("block_size")
        batch_size = get_int_config_value("batch_size")
        vocab_size = get_int_config_value("vocab_size")

        log.debug("Create ML Input...")
        ml_input = model.create_ml_input(train_batch[0])
        self.assertEqual(ml_input.size(dim=0), batch_size)
        self.assertEqual(ml_input.size(dim=1), block_size)
        self.assertEqual(ml_input.size(dim=2), model.tree.depth)
        self.assertEqual(ml_input.size(dim=3), vocab_size)
        log.debug("Done!")

        for b_idx in range(batch_size):
            sample = train_batch[0][b_idx].tolist()
            for t_idx in range(len(sample)):
                for s_idx in range(model.tree.depth):
                    startIdx = t_idx-s_idx
                    if (startIdx >=0):
                        sub_sequence = sample[startIdx:t_idx]
                        _, children = self.rtree.getRandomFrequency(sub_sequence) 
                        for v_idx in range(vocab_size):
                            if (v_idx in children.keys()):
                                self.assertEqual(ml_input[b_idx,t_idx,s_idx,v_idx], children.get(v_idx)[0])
                            else:
                                self.assertEqual(ml_input[b_idx,t_idx,s_idx,v_idx], get_float_config_value('treemodel_zero_value'))
                    else:
                        for v_idx in range(vocab_size):
                            self.assertEqual(ml_input[b_idx,t_idx,s_idx,v_idx], get_float_config_value('treemodel_zero_value'))
    
    def test_forward(self):
        from environment import log
        log.debug('TEST FORWARD')
        from data import DataLoader
        from treemodel import TokenTreeModel
        model = TokenTreeModel(initZeroValue=0.0001)
        loader = DataLoader(block_size=model.tree.depth)
       
        train_batch = loader.batch()

        block_size = model.tree.depth
        batch_size = get_int_config_value("batch_size")
        vocab_size = get_int_config_value("vocab_size")

        log.debug("Do Forward...")
        result = model(train_batch[0])
        self.assertEqual(len(result.size()), 3)
        self.assertEqual(result.size(dim=0), batch_size)
        self.assertEqual(result.size(dim=1), block_size)
        self.assertEqual(result.size(dim=2), vocab_size)

        log.debug("Do Forward...")
        result = model(train_batch[0], inference=True)
        self.assertEqual(len(result.size()), 3)
        self.assertEqual(result.size(dim=0), batch_size)
        self.assertEqual(result.size(dim=1), 1)
        self.assertEqual(result.size(dim=2), vocab_size)
        
        log.debug("Done!")

        

        
    
    def test_cross_entropy(self):
        pass

        
if __name__ == '__main__':
    unittest.main()