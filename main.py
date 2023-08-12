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

def do_init(args):
    log.debug('do_init')

def do_tokenize(args):
    if (args == None):
        log.error("Wrong number of arguments for command tokenize")
    else:
        tokenizer = Tokenizer(args[0])
        tokenizer.tokenize()

if (command == 'init'):
    do_init(args)
elif (command == 'tokenize'):
    do_tokenize(args)


