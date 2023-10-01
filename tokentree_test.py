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

    def createRandomTestData(self, ml, maxTokenListLength, startDepth = 1, stopDepth = 0):
        if (len(self.tokens) == 0):
            number = randint(ml, maxTokenListLength)
            for i in range(number):
                self.tokens.append(randint(0,9))
        maxlength = ml
        if stopDepth == 0:
            stopDepth = ml
        self.frequences = self.frequencies
        self.sequences = self.sequencies
        for idx, _ in enumerate(self.tokens):
            maxl = min(maxlength, len(self.tokens)-idx)
            sequence = self.tokens[idx:idx+maxl]
            for idx2, _ in enumerate(sequence):
                subsequence = sequence[0:idx2+1]
                if (len(subsequence) >= startDepth and len(subsequence) <= stopDepth):
                    self.updateRandomFrequences(self.frequences, subsequence)
                    self.sequences.append(subsequence)
        
    def test_tree_4(self):
        for _ in range(50):
            self.tokens = []
            self.frequencies = {}
            self.sequencies = []
            self.createRandomTestData(5,50)
            tree = TokenTree('testtree.bin', 'w')
            tree.initFirstLevel(10)
            for sequence in self.sequences:
                tree.insertOrUpdateToken(sequence)
            tree.close()
            tree = TokenTree('testtree.bin', 'r')
            
            count = self.countRandomEntries(self.frequencies)
            for token in range(10):
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
            remove('testtree.bin')
    
    def test_tree_5(self):
        for _ in range(25):
            self.tokens = []
            self.frequencies = {}
            self.sequencies = []
            self.createRandomTestData(7,70, startDepth=1, stopDepth=5)
            tree = TokenTree('testtree.bin', 'w')
            tree.initFirstLevel(10)
            for sequence in self.sequences:
                tree.insertOrUpdateToken(sequence)
            tree.close()
            self.createRandomTestData(7,70, startDepth=6, stopDepth=7)
            tree = TokenTree('testtree.bin', 'w')
            for sequence in self.sequences:
                if (len(sequence)>=6):
                    tree.insertOrUpdateToken(sequence)
            tree.close()
            tree = TokenTree('testtree.bin', 'r')
            count = self.countRandomEntries(self.frequencies)
            for token in range(10):
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
            remove('testtree.bin')

    
    

        

        
        


if __name__ == '__main__':
    unittest.main()
