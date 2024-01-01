import unittest

from environment import initEnv
from shutil import rmtree
import numpy as np
from os.path import join
import torch
from torch.nn import functional as F
import math


class GeneratorTest (unittest.TestCase):

    def setUp(self) -> None:
        initEnv('unittest')
        
    
    def tearDown(self) -> None:
        from environment import workDir
        rmtree(workDir)
    
    def generate_logits_from_probs(self, initial_probs, factor):
        tensor_probs = torch.tensor(initial_probs).view(1,len(initial_probs))
        return torch.log(tensor_probs) + math.log(factor)

    def test_get_next_token_probs(self):
        from environment import log
        from generator import TextGenerator
        
        generator = TextGenerator(prompt=None, startToken=0)
        generator.vocab_size = 5
        
        initial_probs = [0.05, 0.1, 0.2, 0.25,0.4]
        logits = self.generate_logits_from_probs(initial_probs, 3.0)
        
        log.debug("Got Logits: "+str(logits)+"-->"+str(F.softmax(logits, dim = -1)))
        
        probs = generator.get_next_token_probs(temperature=1.0, top_p=1.0, logits=logits)
        log.debug("Got Probs.1: "+str(probs))
        for i in range(5):
            self.assertEqual(initial_probs[i], round(probs[0][i].item(),2))
        
        probs = generator.get_next_token_probs(temperature=1.0, top_p=0.8, logits=logits)
        log.debug("Got Probs.2: "+str(probs))
        self.assertEqual(round(torch.sum(probs).item(),2),1.0)
        for i in range(5):
            if (i > 2):
                self.assertEqual(round(initial_probs[i]/0.65,2), round(probs[0][i].item(),2))
            else:
                self.assertEqual(0.0, probs[0][i].item())
        
        probs = generator.get_next_token_probs(temperature=1.5, top_p=1.0, logits=logits)
        log.debug("Got Probs.2: "+str(probs))
        self.assertEqual(round(torch.sum(probs).item(),2),1.0)
    
    def test_get_next_token_probsTopK(self):
        from environment import log
        from generator import TextGenerator
        
        generator = TextGenerator(prompt=None, startToken=0)
        generator.vocab_size = 5
        
        initial_probs = [0.05, 0.1, 0.2, 0.25,0.4]
        logits = self.generate_logits_from_probs(initial_probs, 3.0)
        
        log.debug("Got Logits: "+str(logits)+"-->"+str(F.softmax(logits, dim = -1)))
        
        probs = generator.get_next_token_probs(temperature=1.0, top_k=5, logits=logits)
        log.debug("Got Probs.1: "+str(probs))
        for i in range(5):
            self.assertEqual(initial_probs[i], round(probs[0][i].item(),2))
        
        probs = generator.get_next_token_probs(temperature=1.0, top_k=2, logits=logits)
        log.debug("Got Probs.2: "+str(probs))
        self.assertEqual(round(torch.sum(probs).item(),2),1.0)
        for i in range(5):
            if (i > 2):
                self.assertEqual(round(initial_probs[i]/0.65,2), round(probs[0][i].item(),2))
            else:
                self.assertEqual(0.0, probs[0][i].item())
        
        probs = generator.get_next_token_probs(temperature=1.5, top_p=1.0, logits=logits)
        log.debug("Got Probs.2: "+str(probs))
        self.assertEqual(round(torch.sum(probs).item(),2),1.0)
        
    

if __name__ == '__main__':
    unittest.main()

