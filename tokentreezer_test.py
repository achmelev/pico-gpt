import unittest

from environment import initEnv
from shutil import rmtree
from os.path import isdir
from random import randint
import numpy as np
from tokentree_test import RandomTokenTree

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
    
   
    
    def test_generate_token_tree2(self):
        from environment import log, workDir, get_int_config_value
        from tokentreezer import TokenTreezer
        from tokentree import TokenTree
        rtree = RandomTokenTree()

        rtree.createRandomTestData(5,50, vocab_size=get_int_config_value('vocab_size'))
        np_tokens = np.array(rtree.tokens, dtype=np.uint16)
        trf = open(workDir+"train.bin",'wb')
        np_tokens.tofile(trf)
        trf.close()

        tokentreezer = TokenTreezer()
        tokentreezer.write(5)
        tree = TokenTree(workDir+"tokentree.bin",'r')
        self.assertEqual(tree.depth,5)
        count = rtree.countRandomEntries()+rtree.countTokensNotInFirstLevel(get_int_config_value('vocab_size'))
        self.assertEqual(count, tree.size)
        for sequence in rtree.sequences:
            frequency, children = rtree.getRandomFrequency(sequence)
            node = tree.getNode(sequence)
            self.assertTrue(node != None)
            self.assertEqual(sequence[-1:][0], node.token)
            self.assertEqual(frequency, node.count)
            childrenNodes = tree.getNodesChildren(sequence)
            self.assertTrue(childrenNodes != None)
            self.assertEqual(len(children), len(childrenNodes))
            for token in children.keys():
                frequency1 = children[token][0]
                frequency2 = childrenNodes[token].count
                self.assertEqual(frequency1, frequency2)
        tree.close()
        rtree.createRandomTestData(10,50, startDepth=6, stopDepth=10)
        tokentreezer = TokenTreezer()
        tokentreezer.write(10)
        tree = TokenTree(workDir+"tokentree.bin",'r')
        self.assertEqual(tree.depth,10)
        count = rtree.countRandomEntries()+rtree.countTokensNotInFirstLevel(get_int_config_value('vocab_size'))
        self.assertEqual(count, tree.size)
        for sequence in rtree.sequences:
            frequency, children = rtree.getRandomFrequency(sequence)
            node = tree.getNode(sequence)
            self.assertTrue(node != None)
            self.assertEqual(sequence[-1:][0], node.token)
            self.assertEqual(frequency, node.count)
            childrenNodes = tree.getNodesChildren(sequence)
            self.assertTrue(childrenNodes != None)
            self.assertEqual(len(children), len(childrenNodes))
            for token in children.keys():
                frequency1 = children[token][0]
                frequency2 = childrenNodes[token].count
                self.assertEqual(frequency1, frequency2)
        tree.close()
        

        

if __name__ == '__main__':
    unittest.main()
