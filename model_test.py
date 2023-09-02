import unittest
from environment import initEnv, get_int_config_value
from shutil import rmtree

class ModelTest (unittest.TestCase):

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
    
    def test_forward(self):
        from environment import log
        log.debug('TEST MODEL FORWARD')
        from data import DataLoader
        from model import GPT
        loader = DataLoader()
        model = GPT()
        train_batch = loader.batch()

        block_size = get_int_config_value("block_size")
        batch_size = get_int_config_value("batch_size")
        vocab_size = get_int_config_value("vocab_size")

        logits = model(train_batch[0])

        self.assertEqual(logits.size(dim=0), batch_size)
        self.assertEqual(logits.size(dim=1), block_size)
        self.assertEqual(logits.size(dim=2), vocab_size)

        logits = model(train_batch[0], inference=True)

        self.assertEqual(logits.size(dim=0), batch_size)
        self.assertEqual(logits.size(dim=1), 1)
        self.assertEqual(logits.size(dim=2), vocab_size)

        
        
    
if __name__ == '__main__':
    unittest.main()