from environment import log, device, get_int_config_value, get_float_config_value,get_bool_config_value, workDir
from tokenizer import Tokenizer
from model import GPT, print_config


import torch
from torch.nn import functional as F
from os.path import isfile
from ngrams import Ngrams, hasNgrams

class Score:

    def __init__(self, prompt = None, startToken = None) -> None:

        #Parameters
        self.block_size = get_int_config_value('block_size')

        #Tokenizer
        self.tokenizer = Tokenizer()
        self.vocab_loaded = False
        if (self.tokenizer.has_vocab()):
            self.tokenizer.load_vocab()
            self.vocab_loaded = True
        
        self.startToken = self.tokenizer.vocab_map['<end/>']
        
        #Model
        self.model = GPT()
        self.model.to(device)
        print_config()
        self.model_file = workDir+"model_dict.bin"
        if (isfile(self.model_file)):
            log.info("Loading model from "+self.model_file)
            self.model.load_state_dict(torch.load(self.model_file, map_location = torch.device(device)))
        self.model.eval()

        #StartIndex 
        self.useStartIndex = get_bool_config_value("use_start_index")
    
    def tokenlist_to_tensor(self, tokenlist):
        return torch.tensor()
    
    @torch.no_grad
    def calculate_loss(self, tokens):
        if tokens[0] != self.startToken:
            input_tokens = [self.startToken] + tokens
        else:
            input_tokens = tokens
        
        input_tokens_tensor = torch.tensor(input_tokens,dtype=torch.int64)
        input_tokens_tensor = input_tokens_tensor[None, :]
        logits = self.model(input_tokens_tensor)
        targets = input_tokens_tensor[:,1:]
        loss_logits = logits[:,:-1,:]
        loss = F.cross_entropy(loss_logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss.item()
    
    def calculate(self, text):
        text_tokens = self.tokenizer.tokenize_text(text)
        tokens = [self.tokenizer.vocab_map[t] for t in text_tokens]
        return self.calculate_loss(tokens)







