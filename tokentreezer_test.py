import unittest

from environment import initEnv
from shutil import rmtree
from os.path import isdir

class TokentreezerTest (unittest.TestCase):

    def setUp(self) -> None:
        initEnv('unittest')
    
    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    

    def test_generate_token_tree(self):
        from environment import log, workDir
        from tokenizer import Tokenizer
        from tokentreezer import TokenTreezer
        from tokentree import TokenTree
        log.debug('TEST GENERATE TOKEN TREE')
        tokenizer = Tokenizer()
        input_file = __file__[:__file__.rfind('/')]+'/testdaten/erlkoenig_input.txt'
        tokenizer.generate_vocab(input_file)

        tokenizer.tokenize(input_file)
        tokentreezer = TokenTreezer()
        tokentreezer.write(5)
        tree = TokenTree(workDir+"tokentree.bin",'r')
        self.assertEqual(tree.depth,5)
        tree.close()
        tokentreezer.write(10)
        tree = TokenTree(workDir+"tokentree.bin",'r')
        self.assertEqual(tree.depth,10)
        tree.close()
        

if __name__ == '__main__':
    unittest.main()
