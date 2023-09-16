from os.path import isfile, isdir 
from os import listdir, remove
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
      word_rows = []
      lines = self.readLinesFromFile(file)
      for line in lines:
         words = self.wordTokenizeText(line)
         if self.areWordsSuitable(words):
            word_rows.append(words)
      return word_rows

   def init_files(self, source):
      #Build file list
      files = []
      if isfile(source):
         files.append(source)
         log.info("Reading "+files[0])
      elif isdir(source):
         listing = listdir(source)
         for f in listing:
            files.append(source+f)
         log.info("Reading from "+str(len(files))+" files  in "+source)
      else:
         raise Exception("unknown file "+source)
      return files
   
   def initialize_bpe(self, file):
      #Word frequenzen und vocab
      files = self.init_files(file)
      self.word_freqs = defaultdict(lambda: 0)
      for i, f in enumerate(files):
         log.info(str(i+1)+'. Importing '+str(file))
         word_rows = self.wordTokenizeFile(f)
         for row in word_rows:
            for word in row:
               if (len(word) == 1 and word in self.punctuation):
                  pass
               else:
                  self.word_freqs[word] =  self.word_freqs[word]+1
      
      self.alphabet = []
      for word in self.word_freqs.keys():
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
      
      def merge_pair(a, b, splits, word_freqs, pair_freqs):
         del pair_freqs[(a,b)]
         for word in word_freqs:
            split = splits[word]
            freq = word_freqs[word]
            newtoken = a+b
            if len(split) == 1:
               continue
            i = 0
            while i < len(split) - 1:
                  if split[i] == a and split[i + 1] == b:
                     #Update pair freqs
                     if (i>0):
                        if (pair_freqs[(split[i], a)] <= freq):
                           del pair_freqs[(split[i], a)]
                        else:
                          pair_freqs[(split[i], a)]-= freq
                        pair_freqs[(split[i], newtoken)]+=freq
                     if (i+2 < len(split)):
                        if (pair_freqs[(b, split[i+2])] <= freq):
                           del pair_freqs[(b, split[i+2])]
                        else:
                           pair_freqs[(b, split[i+2])]-= freq
                        pair_freqs[(newtoken, split[i+2])]+=freq
                     split = split[:i] + [newtoken] + split[i + 2 :]
                  else:
                     i += 1
            splits[word] = split
         return splits
      
      splits = {word: [c for c in word] for word in self.word_freqs.keys()}
      target_vocab_size = get_int_config_value('vocab_size')-1-len(self.punctuation)
      pair_freqs = compute_pair_freqs(splits, self.word_freqs)
      while (len(self.vocab) < target_vocab_size):
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
         merge_pair(best_pair[0], best_pair[1], splits, self.word_freqs, pair_freqs)
   
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
      self.initialize_bpe(files)
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
      tokens = []
      percent = 10
      for idx, line in enumerate(lines):
         if ((idx+1) > percent*len(lines)/100):
            log.debug('Line '+str(idx+1)+"/"+str(len(lines)))
            percent = percent+10
         words = self.tokenize_text(line)
         if (words != None):
            tokenList = ([self.vocab_map[w] for w in words])+[self.vocab_map['<end/>']]
            tokens.append(tokenList)
      return tokens
 
   def tokenize(self, source):
      assert self.vocab_prepared, 'no vocab'

      log.info('Tokenizing...')
      tmpf = open(workDir+'temp.bin', 'wb')
      train_dataset_percent = get_int_config_value('train_dataset_percent')
      tokensCounter = 0
      listLengths = []
      files = self.init_files(source)
      for file in files:
         tokens = self.tokenizeFile(file)
         for tokenList in tokens:
            listLengths.append(len(tokenList))
            tokensCounter = tokensCounter + len(tokenList)
            np_tokens = np.array(tokenList, dtype=np.uint16)
            np_tokens.tofile(tmpf)
      tmpf.close()
      log.info('Done. Got '+str(tokensCounter)+' tokens')

      log.info("Writing train and validation set...")
      switchedToValidation = False
      tmpdata =  np.memmap(workDir+'temp.bin', dtype=np.uint16, mode='r')
      f = open(workDir+'train.bin', 'wb')
      numberOfTokens = tokensCounter
      tokensCounter = 0
      for l in listLengths:
         array = tmpdata[tokensCounter:tokensCounter+l]
         tokensCounter+=l
         if not switchedToValidation:
            if tokensCounter > numberOfTokens*train_dataset_percent/100:
               f.close()
               f = open(workDir+'val.bin', 'wb')
               switchedToValidation = True
         array.tofile(f)
      f.close()
      tmpdata._mmap.close()
      remove(workDir+'temp.bin')
      log.info("Done")


      


      if not switchedToValidation: 
               if tokensCounter > numberOfTokens*train_dataset_percent/100:
                  switchedToValidation = True
  




