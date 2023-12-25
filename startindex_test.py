import unittest

from environment import initEnv
from shutil import rmtree
from numpy import array, uint16


class StartIndexTest(unittest.TestCase):

    def prepare_train_file(self):
        from environment import workDir
        f = open(workDir+"train.bin",'wb')
        self.values = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
        arr =  array(self.values, dtype = uint16)
        f.write(arr.tobytes())
        f.close()
    
    def prepare_val_file(self):
        from environment import workDir
        f = open(workDir+"val.bin",'wb')
        self.values_val = [0,1,2,3,4,5,6,7,8,9,0,1,2,3]
        arr =  array(self.values_val, dtype = uint16)
        f.write(arr.tobytes())
        f.close()

    def setUp(self) -> None:
        initEnv('unittest')
        self.prepare_train_file()
        self.prepare_val_file()
        
    
    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    
    def test_generate(self):
        from startindex import StartIndex
        index = StartIndex(readonly=False)
        index.generate(2)
    
    def test_right_padding(self):
        from startindex import StartIndex
        index = StartIndex(readonly=False)
        index.generate(2)
        index = StartIndex(readonly=True, rightPadding=2)
        self.assertEqual(index.length, 3)
        self.assertEqual(index.length_val, 1)
        index = StartIndex(readonly=True, rightPadding=5)
        self.assertEqual(index.length, 3)
        self.assertEqual(index.length_val, 1)

        index = StartIndex(readonly=False)
        index.generate(8)
        index = StartIndex(readonly=True, rightPadding=5)
        self.assertEqual(index.length, 2)
        self.assertEqual(index.length_val, 1)
    
    def test_get_random_pos(self):
        from startindex import StartIndex
        index = StartIndex(readonly=False)
        index.generate(2)
        index = StartIndex(readonly=True)
        for idx in range(10):
            pos = index.getRandomPos()
            self.assertEqual(self.values[pos], 2)
        for idx in range(10):
            pos = index.getRandomPos(train=False)
            self.assertEqual(self.values_val[pos], 2)
        
        result = index.getRandomPos(count=10)
        self.assertEqual(10, len(result))
        for idx in range(10):
            self.assertEqual(self.values[result[idx]], 2)
        result = index.getRandomPos(count=10, train=False)
        self.assertEqual(10, len(result))
        for idx in range(10):
            self.assertEqual(self.values_val[result[idx]], 2)

        
    
    
        
if __name__ == '__main__':
    unittest.main()