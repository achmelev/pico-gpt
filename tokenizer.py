from os.path import isfile, isdir 
from os import listdir
from environment import log
from re import compile, findall


class Tokenizer:
   def  __init__(self, source):
      self.files = []
      self.word_pattern = compile(r'\w[\w\’]*|[.,!?]')
      if isfile(source):
         self.files.append(source)
         log.info("Reading "+self.files[0])
      elif isdir(source):
         files = listdir(source)
         self.files.extend(files)
         log.info("Reading from "+str(len(self.files))+" files  in "+source)
      else:
         raise Exception("unknown file "+source)
   
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

   def tokenize(self):
      self.word_rows = []
      for file in self.files:
         self.wordTokenizeFile(file)
   
   def getWordRows(self):
      return self.word_rows
