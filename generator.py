from environment import log, device, get_int_config_value, get_float_config_value, workDir
from tokenizer import Tokenizer
from model import GPT

import torch
from torch.nn import functional as F
from os.path import isfile

class TextGenerator:

    def __init__(self, prompt) -> None:

        #Parameters
        self.prompt = prompt
        self.block_size = get_int_config_value('block_size')
        self.temperature = get_float_config_value('temperature')
        self.top_k = get_int_config_value('top_k')
        self.max_line_length = get_int_config_value("max_line_length")
        self.max_words = get_int_config_value("max_words")

        #Tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab()
        #Model
        self.model = GPT()
        self.model_file = workDir+"model_dict.bin"
        if (isfile(self.model_file)):
            log.info("Loading model from "+self.model_file)
            self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()
        
        #Context
        tokens = self.tokenizer.tokenize_text(prompt)
        start_ids = [self.tokenizer.vocab_map[t] for t in tokens]
        self.ctx = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])



    @torch.no_grad()
    def generate_token(self) -> str:
        # if the sequence context is growing too long we must crop it at block_size
        self.ctx = self.ctx if self.ctx.size(1) <= self.block_size else self.ctx[:, -self.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits = self.model(self.ctx)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / self.temperature
        # optionally crop the logits to only the top k options
        v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        self.ctx = torch.cat((self.ctx, idx_next), dim=1)

        return self.tokenizer.vocab[idx_next]
    
    def generate_console(self):
        print("#####################################################")
        token = None
        line_length = len(self.prompt)
        words_counter = 0
        print(self.prompt,end="")
        while (words_counter < self.max_words):
            token = self.generate_token()
            if token == '<end/>':
                print("\n")
            elif token[0] == '#':
                if (line_length == 0):
                    print(token[1:], end="")
                    line_length = line_length + len(token) -1
                else:
                    print(' '+token[1:], end="")
                    line_length = line_length + len(token)
                words_counter = words_counter +1
            else:
                print(token,end="")

            if (token != '<end/>'):
                line_length = line_length + len(token)
                if (line_length > self.max_line_length):
                    print("")
                    line_length = 0
        print("")
        print("#####################################################")




