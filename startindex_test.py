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

    def setUp(self) -> None:
        initEnv('unittest')
        self.prepare_train_file()
        
    
    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    


    def test_generate(self):
        self.prepare_train_file()
        from startindex import StartIndex
        index = StartIndex(readonly=False)
        index.generate(2)
    
    def test_get_random_pos(self):
        self.prepare_train_file()
        from startindex import StartIndex
        index = StartIndex(readonly=False)
        index.generate(2)
        index = StartIndex(readonly=True)
        for idx in range(10):
            pos = index.getRandomPos()
            self.assertEqual(self.values[pos], 2)

        
    
    
        
if __name__ == '__main__':
    unittest.main()