from environment import log, workDir, get_int_config_value
from os.path import isfile, isdir 
from numpy import memmap, uint16
from tokenizer import Tokenizer
import torch
from random import randint

class DisPressGenerator:

    def __init__(self, size, max_length):
        assert isfile(workDir+'train.bin'), 'Missing train data file'
        self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab()

        self.context_size = size
        self.depth = size+1
        self.max_length = max_length
        self.max_words = get_int_config_value("max_words")
        self.max_line_length = get_int_config_value("max_line_length")

        self.stats = {}
        self.start_pos = []
        self.words_counter = 0

        self.vocab_size = get_int_config_value('vocab_size')

    def generate_stats(self):
        log.info("Generate stats...")
        end_token = self.tokenizer.vocab_map['<end/>']
        for idx in range(self.max_length-self.depth-10):
            if (idx%10000 == 0):
                log.debug(idx)
            chunk = self.train_data[idx:idx+self.depth]
            if (chunk[0] == end_token):
                self.start_pos.append(idx)
            chunk_tuple = tuple(chunk.tolist())
            if chunk_tuple in self.stats:
                self.stats[chunk_tuple]+=1
            else:
                self.stats[chunk_tuple] = 1
        log.info("Done")
    
    def get_next_token_probs(self, chunk):
        result = torch.zeros(self.vocab_size)
        for t in range(self.vocab_size):
            chunk_tuple = tuple(chunk+[t])
            if (chunk_tuple in self.stats):
                result[t] = float(self.stats[chunk_tuple])
        return result

    def generate_token(self) -> str:
        #Get Probs
        probs = self.get_next_token_probs(self.ctx)
        sum = torch.sum(probs)
        if (sum.item() > 1.0):
            self.tokens_generated_random+=1
        
        if (sum.item() == 0.0):
            return None
        else:
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            self.ctx = self.ctx[1:]+[idx_next.item()]

            return self.tokenizer.vocab[idx_next]

    def prepare(self):
        idx = randint(0, len(self.start_pos)-1)
        self.ctx = self.train_data[self.start_pos[idx]:self.start_pos[idx]+self.context_size].tolist()
        self.start_tokens = [self.tokenizer.vocab[t] for t in self.ctx]


    def generate_console(self):
        log.info("Generating...")
        print("#####################################################")
        token = None
        line_length = 0
        words_counter = 0
        current_word = ""
        prompt_counter = 0
        token_counter = 0
        self.tokens_generated_random = 0
        while (True):
            # Get next token
            if (prompt_counter < len(self.start_tokens)):
                token = self.start_tokens[prompt_counter]
                prompt_counter+=1
            else:
                token = self.generate_token()
            token_counter+=1
            if (token == None):#Not found, reset
                self.prepare()
                prompt_counter = 0
            elif token == '<end/>':
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
        log.info("Done! Generated "+str(words_counter)+" words, "+str(token_counter)+" tokens, "+str(self.tokens_generated_random)+" randomly")




    
    def generate(self):
        self.generate_stats()
        self.prepare()
        self.generate_console()
    

    



            

