import unittest

from environment import initEnv
from shutil import rmtree


class TokenizerTest (unittest.TestCase):

    def setUp(self) -> None:
        print("SETUP")
        initEnv('unittest')
    
    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    
    def test_vocab(self):
        from environment import log
        from tokenizer import Tokenizer
        tokenizer1 = Tokenizer()
        input_file = __file__[:__file__.rfind('/')]+'/testdaten/erlkoenig_input.txt'
        tokenizer1.generate_vocab(input_file)
        tokenizer2 = Tokenizer()
        tokenizer2.load_vocab()

        self.assertEqual(tokenizer1.alphabet, tokenizer2.alphabet)
        self.assertEqual(tokenizer1.merges, tokenizer2.merges)
        self.assertEqual(tokenizer1.vocab, tokenizer2.vocab)

if __name__ == '__main__':
    unittest.main()

