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
    


    def test_generate(self):
        self.prepare_train_file()
        for idx in range(2):
            from ngrams import Ngrams
            ngrams = Ngrams(readonly=False, index=idx)
            ngrams.generate()
            ngrams.close()

    def test_print_stats(self):
        self.prepare_train_file()
        from ngrams import Ngrams
        for idx in range(2):
            from ngrams import Ngrams
            ngrams = Ngrams(readonly=False, index=idx)
            ngrams.generate()
            ngrams.close()
        ngrams = Ngrams(readonly=True)
        ngrams.print_stats()
        ngrams.close()
    
    def test_get_nexts(self):
        self.prepare_train_file()
        from ngrams import Ngrams
        for idx in range(2):
            from ngrams import Ngrams
            ngrams = Ngrams(readonly=False, index=idx)
            ngrams.generate()
            ngrams.close()
        ngrams = Ngrams(readonly=True)
        values = [3,4,5,6,7]
        result = ngrams.get_ngram_nexts(values)
        self.assertEqual(3, len(result))
        for i in result:
            self.assertEqual(8,i)
    
    def test_get_coverage(self):
        self.prepare_train_file()
        from ngrams import Ngrams
        for idx in range(2):
            from ngrams import Ngrams
            ngrams = Ngrams(readonly=False, index=idx)
            ngrams.generate()
            ngrams.close()
        ngrams = Ngrams(readonly=True)
        tokens = [1,2,3,4,5,6,7,0,1,2,3,4]
        result = ngrams.get_ngram_coverage(tokens)
        self.assertEqual(len(tokens), result)
        tokens = [9,1,2,3,4,5,6,7,9,8,0,1,2,3,4,5,8]
        result = ngrams.get_ngram_coverage(tokens)
        self.assertEqual(len(tokens)-4, result)
        
        
    
    
        
if __name__ == '__main__':
    unittest.main()