import unittest

from environment import initEnv
from shutil import rmtree


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

        





if __name__ == '__main__':
    unittest.main()

