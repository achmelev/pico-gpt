from os.path import isfile, isdir 
from os import listdir
from environment import log
from environment import get_config_value, get_int_config_value,workDir
from re import compile, findall
from collections import defaultdict


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
   
   def wordTokenizeText(self, text):
      words = []
      for word in findall(self.word_pattern, text.replace('_','')):
         if (len(word) == 1 and word in self.punctuation):
            words.append(word)
         else:
            words.append('#'+word)
      return words
   
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
         self.word_rows.append(self.wordTokenizeText(current_text))
    
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
               self.word_freqs['#'+word] =  self.word_freqs['#'+word]+1
               words.append(word)
      
      self.alphabet = []
      for word in words:
         for letter in word:
            if letter not in self.alphabet:
                  self.alphabet.append(letter)
      self.alphabet.sort() 
      log.info('Got '+str(len(self.word_freqs))+" words, with "+str(len(self.alphabet))+" chars alphabet")
      self.vocab = self.alphabet.copy()
      self.vocab.append('#')

      
   def bpe(self):
      self.initialize_bpe()
      self.merges = []
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
      assert self.has_words, 'No words available'
      log.info('Generating vocab with '+str(get_int_config_value('vocab_size'))+' tokens...')
      self.read_words(files)
      self.bpe()
      #Append special tokens
      for ch in self.punctuation:
         self.vocab.append(ch)
      self.vocab.append('<end/>')
      log.info('Done. Executed '+str(len(self.merges))+' merges')
      self.verify_vocab()
      self.write_vocab()
      self.setWordPatternFromAlphabet()
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
      self.vocab = []
      for line in lines:
         self.vocab.append(line[:-1])
      log.info('Got '+str(len(self.vocab))+' tokens')
      #Read merges from file
      log.info('Reading merges from '+(workDir+'merges.txt'))
      f = open(workDir+'merges.txt', 'r',encoding='utf8')
      lines = f.readlines()
      self.merges = []
      for line in lines:
         merge = tuple(line[:-1].split(':'))
         self.merges.append(merge)
      log.info('Got '+str(len(self.merges))+' merges')
      self.verify_vocab()
      f.close()
      self.setWordPatternFromAlphabet()
      self.vocab_prepared = True
   
   def tokenize_text(self, text):
      assert self.vocab_prepared, 'no vocab'
      words = self.wordTokenizeText(text)
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
   
   def tokenizeFile(self, file, vocab_map):
      log.info('Tokenizing: '+file)
      f = open(file,'r',encoding='utf8')
      lines = f.readlines()
      current_text = ""
      result = []
      for idx, line in enumerate(lines):
         if ((idx+1)%100 == 0):
            log.debug(str(idx+1)+'/'+str(len(lines)))
         line = line[:len(line)-1]
         if (len(line) > 0):
            current_text+=' '+line
         else:
            if (len(current_text) > 0):
               words = self.tokenize_text(current_text)
               result = result + ([vocab_map[w] for w in words])+[vocab_map['<end/>']]
               current_text = ''
      if (len(current_text) > 0):
         words = self.tokenize_text(current_text)
         result = result + ([vocab_map[w] for w in words])+[vocab_map['<end/>']]

      return result
   
   def tokenize(self, source):
      assert self.vocab_prepared, 'no vocab'
      vocab_map = {value: index for index, value in enumerate(self.vocab)}
      files = self.init_files(source)
      for file in files:
         tokens = self.tokenizeFile(file, vocab_map)




