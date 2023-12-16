from environment import log, workDir, device
from os.path import isfile, isdir 
from numpy import memmap, uint16
from tokenizer import Tokenizer

class TokenStats:

    def __init__(self):
        assert isfile(workDir+'train.bin'), 'Missing train data file'
        assert isfile(workDir+'val.bin'), 'Missing validation data file'
        self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
        self.val_data = memmap(workDir+'val.bin', dtype=uint16, mode='r')

        self.counter_words = 0
        self.counter_tokens = 0
        self.counter_endtokens = 0
        self.counter_punctuation = 0
        self.vocab = []

        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab()

    def update_stats(self, word):
        if len(word) > 0:
            if (word[0] == '#'):
                word = word[1:]
            self.counter_words+=1
            if (word not in self.vocab):
                self.vocab.append(word)

    def generate_from_data(self, data, title):
        log.info('Reading from '+title)
        currentWord = ""
        for idx in range(len(data)):
            token = self.tokenizer.vocab[data[idx]]
            self.counter_tokens+=1
            if (self.max_tokens >0 and self.counter_tokens >= self.max_tokens):
                return
            if (self.counter_tokens%10000 == 0):
                log.debug(self.counter_tokens)
            if (token[0] == '#'):#Beginn des nächsten Wortes
                self.update_stats(currentWord)
                currentWord = token
            elif (len(token) == 1 and token in self.tokenizer.punctuation):#Punctuation
                 self.update_stats(currentWord)
                 currentWord = ""
                 self.counter_punctuation+=1
            elif (token == "<end/>"):#End token
                self.update_stats(currentWord)
                currentWord = ""
                self.counter_endtokens+=1
            else: #Zwischentoken
                currentWord+=token
    
    def generate(self, max_tokens = -1):
        self.max_tokens = max_tokens
        self.generate_from_data(self.train_data,'train file')
        if (self.max_tokens >0 and self.counter_tokens >= self.max_tokens):
            return
        self.generate_from_data(self.train_data,'val file')
    
    def print(self):
        print("########################################################")
        print('Number of tokens: ',self.counter_tokens)
        print('Number of words: ',self.counter_words)
        print('Number of punctuation chars: ',self.counter_punctuation)
        print('Number of end tokens: ',self.counter_endtokens)
        print('Vocab size: ',len(self.vocab))
        print('Vocab: ', self.vocab)
        print("########################################################")



            

