import unittest
from environment import initEnv, get_int_config_value
from shutil import rmtree
from torch.nn import functional as F

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

        self.assertEqual(len(logits.size()), 3)
        self.assertEqual(logits.size(dim=0), batch_size)
        self.assertEqual(logits.size(dim=1), block_size)
        self.assertEqual(logits.size(dim=2), vocab_size)

        logits = model(train_batch[0], inference=True)

        self.assertEqual(len(logits.size()), 3)
        self.assertEqual(logits.size(dim=0), batch_size)
        self.assertEqual(logits.size(dim=1), 1)
        self.assertEqual(logits.size(dim=2), vocab_size)
    
    def test_cross_entropy(self):

        from environment import log
        log.debug('TEST CROSS ENTROPY')

        from data import DataLoader
        from model import GPT
        loader = DataLoader()
        model = GPT()
        train_batch = loader.batch()

        logits = model(train_batch[0])
        targets = train_batch[1]

        inputs = logits.view(-1, logits.size(-1))
        results = targets.view(-1)
        log.debug("Input shape: "+str(inputs.size()))
        log.debug("Result shape: "+str(results.size()))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        log.debug("Result = "+str(loss))

        
        

        
        
    
if __name__ == '__main__':
    unittest.main()