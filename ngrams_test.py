import unittest

from environment import initEnv
from shutil import rmtree
from numpy import array, uint16


class NgramsTest(unittest.TestCase):

    def prepare_train_file(self):
        from environment import workDir
        f = open(workDir+"train.bin",'wb')
        values = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
        arr =  array(values, dtype = uint16)
        f.write(arr.tobytes())
        f.close()

    def setUp(self) -> None:
        initEnv('unittest')
        self.prepare_train_file()
        
    
    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    
    def test_initdb(self):
        from ngrams import Ngrams
        ngrams = Ngrams(readonly=False)
        ngrams.initdb()
        ngrams.close()

    def test_generate(self):
        self.prepare_train_file()
        from ngrams import Ngrams
        ngrams = Ngrams(readonly=False)
        ngrams.initdb()
        ngrams.generate()
        ngrams.close()
    
    def test_count(self):
        self.prepare_train_file()
        from ngrams import Ngrams
        ngrams = Ngrams(readonly=False)
        ngrams.initdb()
        ngrams.generate()
        ngrams.close()
        ngrams = Ngrams()
        value = ngrams.count_ngram(array([1,2,3,4,5], dtype = uint16))
        self.assertEqual(3,value)
    
    def test_count_start_pos(self):
        self.prepare_train_file()
        from ngrams import Ngrams
        ngrams = Ngrams(readonly=False)
        ngrams.initdb()
        ngrams.generate()
        ngrams.close()
        ngrams = Ngrams()
        value = ngrams.count_start_pos()
        self.assertEqual(3,value)
    
    def test_get_start_pos(self):
        self.prepare_train_file()
        from ngrams import Ngrams
        ngrams = Ngrams(readonly=False)
        ngrams.initdb()
        ngrams.generate()
        ngrams.close()
        ngrams = Ngrams()
        self.assertEqual(0,ngrams.get_start_pos(0))
        self.assertEqual(10,ngrams.get_start_pos(1))
        self.assertEqual(20,ngrams.get_start_pos(2))
        
if __name__ == '__main__':
    unittest.main()