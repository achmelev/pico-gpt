import unittest

from environment import initEnv
from shutil import rmtree
from os.path import isdir
from random import randint
import numpy as np

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
    
    def countRandomEntries(self, frequencies):
        currentDict = frequencies
        result = 0
        for key in currentDict.keys():
            result+=1
            result+=self.countRandomEntries(currentDict[key][1])
        
        return result

    def getRandomFrequency(self, frequencies, sequence):
        currentDict = frequencies
        result = None
        for idx, token in enumerate(sequence):
            if token in  currentDict.keys():
                result = currentDict[token]
                currentDict = result[1]
            else:
                raise Exception('Couldnt find', sequence[:idx+1])
        return result[0], result[1]

    def updateRandomFrequences(self, frequencies, sequence):
        currentDict = frequencies
        for idx, token in enumerate(sequence):
            if (token not in currentDict.keys()):
                assert idx == len(sequence)-1,'intermediate token not found!'
                currentDict[token] = [0,{}]
            if (idx == len(sequence)-1):#Update only last token
                currentDict[token][0]+=1
            currentDict = currentDict[token][1]

    def createRandomTestData(self, ml, maxTokenListLength, startDepth = 1, stopDepth = 0, vocab_size = 10):
        if (len(self.tokens) == 0):
            number = randint(ml, maxTokenListLength)
            for i in range(number):
                self.tokens.append(randint(0,vocab_size-1))
        maxlength = ml
        if stopDepth == 0:
            stopDepth = ml
        for idx, _ in enumerate(self.tokens):
            maxl = min(maxlength, len(self.tokens)-idx)
            sequence = self.tokens[idx:idx+maxl]
            for idx2, _ in enumerate(sequence):
                subsequence = sequence[0:idx2+1]
                if (len(subsequence) >= startDepth and len(subsequence) <= stopDepth):
                    self.updateRandomFrequences(self.frequencies, subsequence)
                    self.sequences.append(subsequence)
    
    def test_generate_token_tree2(self):
        from environment import log, workDir, get_int_config_value
        from tokentreezer import TokenTreezer
        from tokentree import TokenTree
        self.tokens = []
        self.frequencies = {}
        self.sequences = []
        self.createRandomTestData(5,50, vocab_size=get_int_config_value('vocab_size'))
        np_tokens = np.array(self.tokens, dtype=np.uint16)
        trf = open(workDir+"train.bin",'wb')
        np_tokens.tofile(trf)
        trf.close()

        tokentreezer = TokenTreezer()
        tokentreezer.write(5)
        tree = TokenTree(workDir+"tokentree.bin",'r')
        self.assertEqual(tree.depth,5)
        count = self.countRandomEntries(self.frequencies)
        for token in range(get_int_config_value('vocab_size')):
            if token not in self.frequencies.keys():
                count+=1
        self.assertEqual(count, tree.size)
        for sequence in self.sequences:
            frequency, children = self.getRandomFrequency(self.frequencies, sequence)
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
        self.createRandomTestData(10,50, startDepth=6, stopDepth=10)
        tokentreezer = TokenTreezer()
        tokentreezer.write(10)
        tree = TokenTree(workDir+"tokentree.bin",'r')
        self.assertEqual(tree.depth,10)
        count = self.countRandomEntries(self.frequencies)
        for token in range(get_int_config_value('vocab_size')):
            if token not in self.frequencies.keys():
                count+=1
        self.assertEqual(count, tree.size)
        for sequence in self.sequences:
            frequency, children = self.getRandomFrequency(self.frequencies, sequence)
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
