import unittest

from tokentree import TokenTreeNode, TokenTree
from os import remove

class TokenTreeTest(unittest.TestCase):

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
        tree = TokenTree('testtree1.bin', 'w')
        self.assertEqual(1, tree.pageSize)
        tree.appendPage()
        self.assertEqual(2, tree.pageSize)
        tree.close()
        tree = TokenTree('testtree1.bin', 'r')
        self.assertEqual(2, tree.pageSize)
        tree.close()
        remove('testtree1.bin')
    
    def test_tree_2(self):
        tree = TokenTree('testtree2.bin', 'w')
        for token in range(400):
            node = TokenTreeNode()
            node.token = token
            tree.appendNode(node)
        tree.close()
        tree = TokenTree('testtree2.bin', 'w')
        for token in range(400):
            node = tree.readNode(token)
            node.count = token
            tree.writeNode(token, node)
        tree.close()

        tree = TokenTree('testtree2.bin', 'w')
        self.assertEqual(400, tree.size)
        for token in range(400):
            node = tree.readNode(token)
            self.assertEqual(token, node.token)
            self.assertEqual(token, node.count)

        tree.close()
        remove('testtree2.bin')

        
        


if __name__ == '__main__':
    unittest.main()
