import unittest
from environment import initEnv, get_int_config_value
from shutil import rmtree

class DataLoaderTest (unittest.TestCase):

    def setUp(self) -> None:
        initEnv('unittest')
        from environment import log
        from tokenizer import Tokenizer
        log.debug('Preparing data')
        tokenizer = Tokenizer()
        input_file = __file__[:__file__.rfind('/')]+'/testdaten/erlkoenig_input.txt'
        tokenizer.generate_vocab(input_file)
        tokenizer.tokenize(input_file) 
        
    
    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    
    def test_batch(self):
        from environment import log
        log.debug('TEST BATCH')
        from data import DataLoader
        loader = DataLoader()
        train_batch = loader.batch()
        val_batch = loader.batch(train=False)

        block_size = get_int_config_value("block_size")
        batch_size = get_int_config_value("batch_size")

        samples = train_batch[0]
        targets = train_batch[1]
        self.assertEqual(samples.size(dim=0), batch_size)
        self.assertEqual(samples.size(dim=1), block_size)
        self.assertEqual(targets.size(dim=0), batch_size)
        self.assertEqual(targets.size(dim=1), block_size)

        samples = val_batch[0]
        targets = val_batch[1]
        self.assertEqual(samples.size(dim=0), batch_size)
        self.assertEqual(samples.size(dim=1), block_size)
        self.assertEqual(targets.size(dim=0), batch_size)
        self.assertEqual(targets.size(dim=1), block_size)
    
if __name__ == '__main__':
    unittest.main()

