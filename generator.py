from environment import log, device, get_int_config_value, get_float_config_value, workDir
from tokenizer import Tokenizer
from model import GPT, print_config


import torch
from torch.nn import functional as F
from os.path import isfile
from ngrams import Ngrams

class TextGenerator:

    def __init__(self, prompt = None, startToken = None) -> None:

        #Parameters
        self.block_size = get_int_config_value('block_size')
        self.temperature = get_float_config_value('temperature')
        self.top_p = get_float_config_value('top_p')
        self.max_line_length = get_int_config_value("max_line_length")
        self.max_words = get_int_config_value("max_words")
        self.vocab_size = get_int_config_value("vocab_size")

        #Tokenizer
        self.tokenizer = Tokenizer()
        self.vocab_loaded = False
        if (self.tokenizer.has_vocab()):
            self.tokenizer.load_vocab()
            self.vocab_loaded = True
        
        #Model
        self.model = GPT()
        self.model.to(device)
        print_config()
        self.model_file = workDir+"model_dict.bin"
        if (isfile(self.model_file)):
            log.info("Loading model from "+self.model_file)
            self.model.load_state_dict(torch.load(self.model_file, map_location = torch.device(device)))
        self.model.eval()
        
        #Context
        if (prompt != None):
            self.start_tokens = self.tokenizer.tokenize_text(prompt)
            if (self.start_tokens == None):
                self.start_tokens = ['<end/>']
            self.start_ids = [self.tokenizer.vocab_map[t] for t in self.start_tokens]
        else:
            self.start_ids = [startToken]
        
        self.ctx = (torch.tensor(self.start_ids, dtype=torch.long, device=device)[None, ...])

        #Ngrams
        self.ngrams = Ngrams(readonly=True)

    @torch.no_grad()
    def get_next_token_probs(self, logits, temperature, top_p):
        # apply temperature
        assert temperature > 0.0, "Illegal temperature "+str(temperature)
        if (temperature != 1.0):
            logits = logits/temperature
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        assert top_p > 0.0 and top_p <= 1.0, "Illegal top_p "+str(top_p)
        #Sort probs descending (probs doesn't change itself)
        if (top_p < 1.0):
            sorted = torch.sort(probs, descending = True)
            cum_prob = 0.0
            for i in range(self.vocab_size):
                current_prob = sorted[0][0][i].item()
                if (cum_prob + current_prob >= self.top_p and i>0):
                    probs[0][sorted[1][0][i].item()] = 0.0
                cum_prob+=current_prob
            probs = F.normalize(probs, p = 1.0)
        return probs


    @torch.no_grad()
    def generate_token(self) -> str:
        assert self.vocab_loaded, "No Vocab"
        # if the sequence context is growing too long we must crop it at block_size
        self.ctx = self.ctx if self.ctx.size(1) <= self.block_size else self.ctx[:, -self.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits = self.model(self.ctx)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] 

        #Get Probs
        probs = self.get_next_token_probs(logits, self.temperature, self.top_p)
        
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        self.ctx = torch.cat((self.ctx, idx_next), dim=1)

        return self.tokenizer.vocab[idx_next]
    
    def generate_console(self):
        log.info("Generating...")
        print("#####################################################")
        token = None
        line_length = 0
        words_counter = 0
        current_word = ""
        prompt_counter = 0
        token_counter = 0
        result_tokens = []+self.start_ids
        while (True):
            # Get next token
            if (prompt_counter < len(self.start_tokens)):
                token = self.start_tokens[prompt_counter]
                prompt_counter+=1
            else:
                token = self.generate_token()
                result_tokens.append(self.tokenizer.vocab_map[token])
            token_counter+=1
            if token == '<end/>':
                if len(current_word) >0:
                    if (line_length == 0):
                        print(current_word, end="")
                    else:
                        print(" "+current_word, end="")
                if (words_counter > self.max_words):
                    print("")
                    break
                if (token_counter >1):#Nicht wenn <end/> der allererste Token ist.
                    print("\n")#Leerzeile
                line_length = 0
                current_word = ""
            elif token[0] == '#':
                if (line_length == 0):
                    if len(current_word) >0:
                        print(current_word, end="")#Aktuelles Wort
                    line_length = line_length + len(current_word)
                else:
                    if len(current_word)>0:
                        print(' '+current_word, end = "")#Aktuelles Wort
                    line_length = line_length + len(current_word)+1
                current_word = token[1:]
                words_counter = words_counter +1
                if (line_length > self.max_line_length):
                    print("")#Zeilenumbruch
                    line_length = 0
            else:
                current_word = current_word+token
        print("#####################################################")
        log.info("Done! Generated "+str(words_counter)+" words, "+str(token_counter)+" tokens")
        if self.ngrams.active:
            covered_tokens = self.ngrams.get_ngram_coverage(result_tokens)
            log.info("Ngrams coverage: "+str(int(round(float(covered_tokens)*100.0/float(len(result_tokens)))))+"%")




