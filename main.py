from environment import initEnv
from sys import argv

if (len(argv) < 3):
    raise Exception('Called with less as 2 arguments')

initEnv(argv[1])

from environment import log
from tokenizer import Tokenizer

command = argv[2]
args = None
if (len(argv) > 3):
    args = argv[3:]

def do_vocab(args):
    if (args == None):
        log.error("Wrong number of arguments for command vocab")
    else:
        tokenizer = Tokenizer()
        tokenizer.generate_vocab(args[0])

def do_tokenize(args):
    if (args == None):
        log.error("Wrong number of arguments for command vocab")
    else:
        tokenizer = Tokenizer()
        tokenizer.load_vocab()
        result = tokenizer.tokenize(args[0])

def do_tokenizetext(args):
    if (args == None):
        log.error("Wrong number of arguments for command vocab")
    else:
        tokenizer = Tokenizer()
        tokenizer.load_vocab()
        result = tokenizer.tokenize_text(args[0])
        if (result):
            tokenizer.verify_tokens(result)
            log.info("Result: "+str(result))
        else:
            log.info("Too short or no punktuation")

if (command == 'vocab'):
    do_vocab(args)
elif (command == 'tokenize'):
    do_tokenize(args)
elif (command == 'tokenizetext'):
    do_tokenizetext(args)



