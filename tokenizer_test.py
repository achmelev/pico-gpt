import unittest

from environment import initEnv
from shutil import rmtree
import numpy as np
from os.path import join


class TokenizerTest (unittest.TestCase):

    def setUp(self) -> None:
        initEnv('unittest')
        
    
    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    
    def test_vocab(self):
        from environment import log
        from tokenizer import Tokenizer
        log.debug('TEST VOCAB')
        tokenizer1 = Tokenizer()
        input_file = __file__[:__file__.rfind('/')]+'/testdaten/erlkoenig_input.txt'
        tokenizer1.generate_vocab(input_file)
        tokenizer2 = Tokenizer()
        tokenizer2.load_vocab()
        self.assertEqual(tokenizer1.alphabet, tokenizer2.alphabet)
        self.assertEqual(tokenizer1.merges, tokenizer2.merges)
        self.assertEqual(tokenizer1.vocab, tokenizer2.vocab)
    
    def test_tokenize_text(self):
        from environment import log
        from tokenizer import Tokenizer
        log.debug('TEST TOKENIZE TEXT')
        tokenizer = Tokenizer()
        input_file = __file__[:__file__.rfind('/')]+'/testdaten/erlkoenig_input.txt'
        tokenizer.generate_vocab(input_file)

        input_text = "Es lebe die  bürgerliche Revolution!"
        target_text = "Es lebe die bürgerliche evolution!"
        
        tokens = [tokenizer.vocab_map[t] for t in tokenizer.tokenize_text(input_text)]
        text_again = tokenizer.tokens_to_text(tokens)

        self.assertEqual(text_again, target_text)
    
    def test_tokenize(self):
        from environment import log, workDir
        from tokenizer import Tokenizer
        log.debug('TEST TOKENIZE')
        tokenizer = Tokenizer()
        input_file = __file__[:__file__.rfind('/')]+'/testdaten/erlkoenig_input.txt'
        tokenizer.generate_vocab(input_file)

        tokenizer.tokenize(input_file)

        train_data = np.memmap(join(workDir, 'train.bin'), dtype=np.uint16, mode='r')
        val_data = np.memmap(join(workDir, 'val.bin'), dtype=np.uint16, mode='r')

        all_tokens = np.concatenate((train_data, val_data))

        text_again = tokenizer.tokens_to_text(all_tokens)
        compare_file = open(__file__[:__file__.rfind('/')]+'/testdaten/erlkoenig_cleaned.txt','r')
        compare_text = compare_file.read()
        compare_file.close()

        self.assertEqual(len(text_again), len(compare_text))
        self.assertEqual(text_again, compare_text)

if __name__ == '__main__':
    unittest.main()

