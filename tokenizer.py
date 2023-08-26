from os.path import isfile, isdir 
from os import listdir
from environment import log
from environment import get_config_value, get_int_config_value,workDir
from re import compile, findall
from collections import defaultdict
import numpy as np


class Tokenizer:
   def  __init__(self):

      #get regex pattern
      self.punctuation = get_config_value('punctuation_chars')
      wordPatternStr = r'\w[\w\’]*|['+self.punctuation+']'
      self.word_pattern = compile(wordPatternStr)
      self.has_words = False
      self.vocab_prepared = False
   
   def setWordPatternFromAlphabet(self):
      wordPatternStr = '['+self.alphabet+']+|['+self.punctuation+']'
      self.word_pattern = compile(wordPatternStr)
   
   def areWordsSuitable(self, words):
      result = len(words) >= get_int_config_value('min_words_in_sentence')
      return result
   
   def prepare_vocab_map(self):
      self.vocab_map = {value: index for index, value in enumerate(self.vocab)}
      

   def wordTokenizeText(self, text):
      words = []
      for word in findall(self.word_pattern, text.replace('_','')):
         if (len(word) == 1 and word in self.punctuation):
            words.append(word)
         else:
            words.append('#'+word)
      return words
   
   def readLinesFromFile(self, file):
      f = open(file,'r',encoding='utf8')
      lines = f.readlines()
      f.close()
      lines.append('\n')
      current_text = ""
      result = []
      for line in lines:
         line = line[:len(line)-1].strip()
         if (len(line) > 0):
            current_text+=' '+line
         else:
            if (len(current_text) > 0):
               result.append(current_text)    
               current_text = ''
      return result

   def wordTokenizeFile(self, file):
      lines = self.readLinesFromFile(file)
      for line in lines:
         words = self.wordTokenizeText(line)
         if self.areWordsSuitable(words):
            self.word_rows.append(words)

   def init_files(self, source):
      assert not self.has_words, 'Words already loaded'
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
      return files
   
   def read_words(self, source):
      files = self.init_files(source)
      self.word_rows = []
      for file in files:
         self.wordTokenizeFile(file)
      
      self.has_words = True

   def initialize_bpe(self):
      #Word frequenzen und vocab
      self.word_freqs = defaultdict(lambda: 0)
      words = []
      for row in self.word_rows:
         for word in row:
            if (len(word) == 1 and word in self.punctuation):
               pass
            else:
               self.word_freqs[word] =  self.word_freqs[word]+1
               words.append(word)
      
      self.alphabet = []
      for word in words:
         for letter in word:
            if letter not in self.alphabet and not letter == '#':
                  self.alphabet.append(letter)
      self.alphabet.sort() 
      log.info('Got '+str(len(self.word_freqs))+" words, with "+str(len(self.alphabet))+" chars alphabet")
      self.vocab = self.alphabet.copy()
      self.alphabet = ''.join(self.alphabet)
      self.vocab.append('#')
      self.merges = []

      
   def bpe(self):
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
         token = best_pair[0]+best_pair[1]
         self.merges.append((best_pair[0], best_pair[1], token))
         self.vocab.append(token)
         if (len(self.vocab)%25 == 0):
            log.debug('Vocab size: '+str(len(self.vocab))+', merges size: '+str(len(self.merges)))
         merge_pair(best_pair[0], best_pair[1], splits, self.word_freqs)
   
   def has_vocab(self):
      return isfile(workDir+'vocab.txt') and isfile(workDir+'merges.txt') and isfile(workDir+'alphabet.txt')

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
      log.info('Writing merges to '+(workDir+'merges.txt'))
      f = open(workDir+'merges.txt', 'w',encoding='utf8')
      for merge in self.merges:
         f.write(merge[0]+':'+merge[1]+':'+merge[2]+'\n')
      f.close()

   def generate_vocab(self, files):
      assert not self.vocab_prepared, 'vocab already generated/loaded'
      log.info('Generating vocab with '+str(get_int_config_value('vocab_size'))+' tokens...')
      self.read_words(files)
      self.initialize_bpe()
      self.bpe()
      #Append special tokens
      for ch in self.punctuation:
         self.vocab.append(ch)
      self.vocab.append('<end/>')
      log.info('Done. Executed '+str(len(self.merges))+' merges')
      self.verify_vocab()
      self.write_vocab()
      self.setWordPatternFromAlphabet()
      self.prepare_vocab_map()
      self.vocab_prepared = True
   
   def verify_vocab(self):
      log.info('Verifying vocab..')
      assert len(self.vocab) == len(self.alphabet)+1+len(self.merges)+len(self.punctuation)+1
      for i, token in enumerate(self.vocab):
         if i < len(self.alphabet):
            assert token == self.alphabet[i], 'expected '+self.alphabet[i]+', but got '+token
         elif (i==len(self.alphabet)):
            assert token == '#', 'expected #, but got '+token
         elif (i>len(self.alphabet) and i<len(self.alphabet)+1+len(self.merges)):
            merge = self.merges[i-len(self.alphabet)-1]
            assert token == merge[2],'expected '+merge[2]+', but got '+token
         elif (i>=len(self.alphabet)+1+len(self.merges) and i<len(self.vocab)-1):
            assert token == self.punctuation[i-len(self.alphabet)-1-len(self.merges)],'expected '+self.punctuation[i-len(self.alphabet)-1-len(self.merges)]+', but got '+token
         else:
            assert token == '<end/>','expected <end/>, but got '+token
      log.info('Done')

   def load_vocab(self):
      assert not self.vocab_prepared, 'vocab already generated/loaded'
      assert self.has_vocab(), 'No vocab files found!'
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
      f.close()
      self.vocab = []
      for line in lines:
         self.vocab.append(line[:-1])
      log.info('Got '+str(len(self.vocab))+' tokens')
      #Read merges from file
      log.info('Reading merges from '+(workDir+'merges.txt'))
      f = open(workDir+'merges.txt', 'r',encoding='utf8')
      lines = f.readlines()
      f.close()
      self.merges = []
      for line in lines:
         merge = tuple(line[:-1].split(':'))
         self.merges.append(merge)
      log.info('Got '+str(len(self.merges))+' merges')
      self.verify_vocab()
      self.setWordPatternFromAlphabet()
      self.prepare_vocab_map()
      self.vocab_prepared = True
   
   def tokenize_text(self, text):
      assert self.vocab_prepared, 'no vocab'
      words = self.wordTokenizeText(text)
      if not self.areWordsSuitable(words):
         return None
      splits = [[l for l in word] for word in words]
      for merge in self.merges:
         for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == merge[0] and split[i + 1] == merge[1]:
                    split = split[:i] + [merge[2]] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

      return sum(splits, [])
   
   def tokens_to_text(self, tokens):
      assert  not any(t not in range(len(self.vocab)) for t in tokens)

      result = ""
      begin = True
      for t in tokens:
         text_token = self.vocab[t]
         if (text_token in self.punctuation):
            result = result+text_token
         elif(text_token == '<end/>'):
            result = result+'\n\n'
            begin = True
         else:
            if (text_token[0] == '#'):
               if begin:
                  result = result + text_token[1:]
                  begin = False
               else:
                  result = result + ' '+text_token[1:]
            else:
               result = result + text_token
      return result
         
   def verify_tokens(self, tokens):
      assert self.vocab_prepared, 'no vocab'
      log.info('Verifying tokens')
      assert  not any(t not in self.vocab for t in tokens)

   
   def tokenizeFile(self, file):
      log.info('Tokenizing: '+file)
      lines = self.readLinesFromFile(file)
      percent = 10
      for idx, line in enumerate(lines):
         if ((idx+1) > percent*len(lines)/100):
            log.debug('Line '+str(idx+1)+"/"+str(len(lines)))
            percent = percent+10
         words = self.tokenize_text(line)
         if (words != None):
            tokenList = ([self.vocab_map[w] for w in words])+[self.vocab_map['<end/>']]
            self.tokens.append(tokenList)
 
   def tokenize(self, source):
      assert self.vocab_prepared, 'no vocab'
      self.word_rows = None
      self.has_words = False
      files = self.init_files(source)
      self.tokens = []
      for file in files:
         self.tokenizeFile(file)
      
      numberOfTokens = sum(len(row) for row in self.tokens)
      log.info('Got '+str(numberOfTokens)+' tokens')
      log.info('Writing training and validation set...')
      f = open(workDir+'train.bin', 'wb')
      switchedToValidation = False
      train_dataset_percent = get_int_config_value('train_dataset_percent')
      tokensCounter = 0
      for tokenList in self.tokens:
         tokensCounter = tokensCounter + len(tokenList)
         if not switchedToValidation: 
            if tokensCounter > numberOfTokens*train_dataset_percent/100:
               f.close()
               f = open(workDir+'val.bin', 'wb')
               switchedToValidation = True
         np_tokens = np.array(tokenList, dtype=np.uint16)
         np_tokens.tofile(f)
      f.close()
  




