from environment import initEnv, get_int_config_value
from sys import argv

if (len(argv) < 3):
    raise Exception('Called with less as 2 arguments')

initEnv(argv[1])

from environment import log
from tokenizer import Tokenizer
from model import GPT
from generator import TextGenerator
from train import Trainer

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
        log.error("Wrong number of arguments for command tokenize")
    else:
        tokenizer = Tokenizer()
        tokenizer.load_vocab()
        result = tokenizer.tokenize(args[0])

def do_generate(args):
    if (args == None):
        prompt = ""
    else:
        prompt = args[0]
    generator = TextGenerator(prompt)
    generator.generate_console()
       

def do_tokenizetext(args):
    if (args == None):
        log.error("Wrong number of arguments for command tokenizetext")
    else:
        tokenizer = Tokenizer()
        tokenizer.load_vocab()
        result = tokenizer.tokenize_text(args[0])
        if (result):
            tokenizer.verify_tokens(result)
            log.info("Result: "+str(result))
        else:
            log.info("Too short or no punktuation")

def do_train(args):
    if (args == None):
        log.error("Wrong number of arguments for command train")
    else:
        minutes_to_train = int(args[0])
        trainer = Trainer(minutes_to_train)
        trainer.run()
        


if (command == 'vocab'):
    do_vocab(args)
elif (command == 'tokenize'):
    do_tokenize(args)
elif (command == 'tokenizetext'):
    do_tokenizetext(args)
elif (command == 'train'):
    do_train(args)
elif (command == 'generate'):
    do_generate(args)
elif (command == 'train'):
    do_train(args)



