import unittest

from tokentree import TokenTreeNode, TokenTree
from os import remove
from os.path import isfile
from random import randint

class TokenTreeTest(unittest.TestCase):

    def setUp(self) -> None:
        if isfile('testtree.bin'):
            remove('testtree.bin')

    def test_node(self):
        node = TokenTreeNode()
        self.assertEqual(0, node.token)
        self.assertEqual(0, node.count)
        self.assertEqual(0, node.sibling)
        self.assertEqual(0, node.child)
        node.token = 5
        node.count = 10
        node.sibling = 16
        node.child = 25
        self.assertEqual(5, node.token)
        self.assertEqual(10, node.count)
        self.assertEqual(16, node.sibling)
        self.assertEqual(25, node.child)
        node.token = 4009
        node.count = 100005
        node.sibling = 1230006000
        node.child = 267004006
        self.assertEqual(4009, node.token)
        self.assertEqual(100005, node.count)
        self.assertEqual(1230006000, node.sibling)
        self.assertEqual(267004006, node.child)
        node = TokenTreeNode(node.content)
        self.assertEqual(4009, node.token)
        self.assertEqual(100005, node.count)
        self.assertEqual(1230006000, node.sibling)
        self.assertEqual(267004006, node.child)
    
    def test_tree_1(self):
        tree = TokenTree('testtree.bin', 'w')
        self.assertEqual(1, tree.pageSize)
        tree.appendPage()
        self.assertEqual(2, tree.pageSize)
        tree.close()
        tree = TokenTree('testtree.bin', 'r')
        self.assertEqual(2, tree.pageSize)
        tree.close()
    
    def test_tree_2(self):
        tree = TokenTree('testtree.bin', 'w')
        for token in range(400):
            node = TokenTreeNode()
            node.token = token
            tree.appendNode(node)
        tree.close()
        tree = TokenTree('testtree.bin', 'w')
        for token in range(400):
            node = tree.readNode(token)
            node.count = token
            tree.writeNode(node)
        tree.close()

        tree = TokenTree('testtree.bin', 'w')
        self.assertEqual(400, tree.size)
        for token in range(400):
            node = tree.readNode(token)
            self.assertEqual(token, node.token)
            self.assertEqual(token, node.count)
        tree.close()

    def test_tree_3(self):
        tree = TokenTree('testtree.bin', 'w')
        tree.initFirstLevel(50)
        tree.close()
        tree = TokenTree('testtree.bin', 'r')
        for token in range(50):
            node = tree.getNode([token])
            self.assertTrue(node != None)
        tree.close()
        remove('testtree.bin')
    
    def test_tree_3(self):
        tree = TokenTree('testtree.bin', 'w')
        tree.initFirstLevel(50)
        tree.close()
        tree = TokenTree('testtree.bin', 'r')
        for token in range(50):
            node = tree.getNode([token])
            self.assertTrue(node != None)
        tree.close()
        remove('testtree.bin')
    
    def updateRandomFrequences(self, frequencies, sequence):
        currentDict = frequencies
        for idx, token in enumerate(sequence):
            if (token not in currentDict.keys()):
                assert idx == len(sequence)-1,'intermediate token not found!'
                currentDict[token] = [0,{}]
            if (idx == len(sequence)-1):#Update only last token
                currentDict[token][0]+=1
            currentDict = currentDict[token][1]

    def createRandomTestData(self):
        tokens = []
        for i in range(100):
            tokens.append(randint(0,9))
        maxlength = 5
        frequences = {}
        sequences = []
        for idx, _ in enumerate(tokens):
            maxl = min(maxlength, len(tokens)-idx)
            sequence = tokens[idx:maxl]
            for idx2, _ in enumerate(sequence):
                subsequence = sequence[0:idx2+1]
                self.updateRandomFrequences(frequences, subsequence)
                sequences.append(subsequence)
        
        return sequences, frequences


    def test_tree_4(self):
        
        for _ in range(50):
            sequences, frequences = self.createRandomTestData()
            tree = TokenTree('testtree.bin', 'w')
            tree.initFirstLevel(10)
            for sequence in sequences:
                tree.insertOrUpdateToken(sequence)
            tree.close()
            remove('testtree.bin')
        

        
        


if __name__ == '__main__':
    unittest.main()
