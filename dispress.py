from environment import log, workDir, get_int_config_value
from os.path import isfile, isdir 
from numpy import memmap, uint16
from tokenizer import Tokenizer
import torch
from startindex import StartIndex
from ngrams import Ngrams
from random import randint

class DisPressGenerator:

    def __init__(self):
        assert isfile(workDir+'train.bin'), 'Missing train data file'
        self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab()

        self.max_words = get_int_config_value("max_words")
        self.max_line_length = get_int_config_value("max_line_length")

        self.stats = {}
        self.start_pos = []
        self.words_counter = 0

        self.vocab_size = get_int_config_value('vocab_size')

        
        self.ngram_size = self.ngram_size = get_int_config_value('ngram_size')
        self.startindex = StartIndex(readonly=True)
        self.ngrams = Ngrams(readonly=True)



    def generate_token(self) -> str:
        next_tokens = self.ngrams.get_ngram_nexts(self.ctx)
        if (len(next_tokens) == 1):
            idx = 0
        else:
            idx = randint(0,len(next_tokens)-1)
            self.tokens_generated_random+=1
        next_token  = next_tokens[idx]
        self.ctx = self.ctx[1:]+[next_token]
        return self.tokenizer.vocab[next_token]

    def prepare(self):
        idx = self.startindex.getRandomPos()
        self.ctx = self.train_data[idx:idx+self.ngram_size].tolist()
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
        self.prepare()
        self.generate_console()
    

    



            

