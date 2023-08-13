from os.path import isfile, isdir 
from os import listdir
from environment import log
from environment import get_config_value, get_int_config_value,workDir
from re import compile, findall
from collections import defaultdict
from json import loads, dumps


class Tokenizer:
   def  __init__(self):

      #get regex pattern
      self.punctuation = get_config_value('punctuation_chars')
      wordPatternStr = r'\w[\w\’]*|['+self.punctuation+']'
      self.word_pattern = compile(wordPatternStr)
      #log.debug('Word Pattern: '+wordPatternStr)
      
   
   def has_words(self):
      return (self.word_rows != None)
   
   def wordTokenizeFile(self, file):
      f = open(file,'r',encoding='utf8')
      lines = f.readlines()
      current_text = ""
      for line in lines:
         line = line[:len(line)-1]
         if (len(line) > 0):
            current_text+=' '+line
         else:
            if (len(current_text) > 0):
               self.word_rows.append(findall(self.word_pattern, current_text.replace('_','')))
               current_text = ''
      if (len(current_text) > 0):
         self.word_rows.append(findall(self.word_pattern, current_text.replace('_','')))

   def pre_tokenize(self, text):
      #pass
      return None
   
   def read_words(self, source):
      #Build file list
      files = []
      if isfile(source):
         files.append(source)
         log.info("Reading "+files[0])
      elif isdir(source):
         files = listdir(source)
         files.extend(files)
         log.info("Reading from "+str(len(files))+" files  in "+source)
      else:
         raise Exception("unknown file "+source)

      self.word_rows = []
      for file in files:
         self.wordTokenizeFile(file)

   def initialize_bpe(self):
      #Word frequenzen und vocab
      self.word_freqs = defaultdict(lambda: 0)
      for row in self.word_rows:
         for word in row:
            if (len(word) == 1 and word in self.punctuation):
               pass
            else:
               self.word_freqs['#'+word] =  self.word_freqs['#'+word]+1
      
      self.alphabet = []
      for word in self.word_freqs.keys():
         for letter in word:
            if letter not in self.alphabet:
                  self.alphabet.append(letter)
      self.alphabet.sort() 
      
      log.info('Got '+str(len(self.word_freqs))+" words, with "+str(len(self.alphabet))+" chars alphabet")
      self.vocab = self.alphabet.copy()
      self.merges = {}

      
   def bpe(self):
      self.initialize_bpe()
      def compute_pair_freqs(splits, word_freqs):
         pair_freqs = defaultdict(int)
         for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
               continue
            for i in range(len(split) - 1):
                  pair = (split[i], split[i + 1])
                  pair_freqs[pair] += freq
         return pair_freqs
      
      def merge_pair(a, b, splits, word_freqs):
         for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
               continue
            i = 0
            while i < len(split) - 1:
                  if split[i] == a and split[i + 1] == b:
                     split = split[:i] + [a + b] + split[i + 2 :]
                  else:
                     i += 1
            splits[word] = split
         return splits
      
      splits = {word: [c for c in word] for word in self.word_freqs.keys()}
      target_vocab_size = get_int_config_value('vocab_size')-1-len(self.punctuation)
      while (len(self.vocab) < target_vocab_size):
         pair_freqs = compute_pair_freqs(splits, self.word_freqs)
         #Search best pair
         best_pair = ""
         max_freq = None
         for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
               best_pair = pair
               max_freq = freq
         self.merges[best_pair[0]+best_pair[1]] =  best_pair
         self.vocab.append(best_pair[0]+best_pair[1])
         if (len(self.vocab)%25 == 0):
            log.debug('Vocab size: '+str(len(self.vocab)))
         merge_pair(best_pair[0], best_pair[1], splits, self.word_freqs)
   
   def has_vocab(self):
      return isfile(workDir+'vocab.txt') and isfile(workDir+'merges.json') and isfile(workDir+'alphabet.txt')

   def write_vocab(self):
      #Write alphabet to file
      line = ''
      for ch in self.alphabet:
         line+=ch
      log.info('Writing alphabet to '+(workDir+'alphabet.txt'))
      f = open(workDir+'alphabet.txt', 'w',encoding='utf8')
      f.write(line)
      f.close()
      #Write vocab to file
      log.info('Writing vocab to '+(workDir+'vocab.txt'))
      f = open(workDir+'vocab.txt', 'w',encoding='utf8')
      for token in self.vocab:
         f.write(token+'\n')
      f.close()
      #Write merges to file
      log.info('Writing merges to '+(workDir+'merges.json'))
      f = open(workDir+'merges.json', 'w',encoding='utf8')
      f.write(dumps(self.merges))
      f.close()

   def generate_vocab(self):
      if (not self.has_words):
         raise Exception('No words available')
      log.info('Generating vocab with '+str(get_int_config_value('vocab_size'))+' tokens...')
      self.bpe()
      #Append special tokens
      for ch in self.punctuation:
         self.vocab.append(ch)
      self.vocab.append('<end/>')
      log.info('Done. Executed '+str(len(self.merges))+' merges')
      self.write_vocab()
   
   def load_vocab(self):
      if (not self.has_vocab()):
         raise Exception('No vocab files found!')
      #Read alphabet from file
      log.info('Reading alphabet from '+(workDir+'alphabet.txt'))
      f = open(workDir+'alphabet.txt', 'r',encoding='utf8')
      self.alphabet = f.read()
      log.info('Got '+str(len(self.alphabet))+' chars')
      f.close()
      #Read vocab from file
      log.info('Reading vocab from '+(workDir+'vocab.txt'))
      f = open(workDir+'vocab.txt', 'r',encoding='utf8')
      lines = f.readlines()
      self.vocab = []
      for line in lines:
         self.vocab.append(line[:-1])
      log.info('Got '+str(len(self.vocab))+' tokens')
      #Read merges from file
      log.info('Reading merges from '+(workDir+'merges.json'))
      f = open(workDir+'merges.json', 'r',encoding='utf8')
      text = f.read()
      self.merges = loads(text)
      log.info('Got '+str(len(self.merges))+' merges')
      f.close()
