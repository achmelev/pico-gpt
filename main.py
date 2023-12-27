from environment import initEnv, get_int_config_value
from sys import argv

if (len(argv) < 3):
    raise Exception('Called with less as 2 arguments')

initEnv(argv[1])

from environment import log
from tokenizer import Tokenizer
from model import GPT, print_config
from generator import TextGenerator
from train import Trainer
from downloader import EnvDownloader
from profiler import profile_run
from tokenstats import TokenStats
from ngrams import Ngrams
from dispress import DisPressGenerator
from startindex import StartIndex
from score import Score

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
    generator = TextGenerator(prompt = prompt)
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

def do_download(args):
    if (args == None):
        log.error("Wrong number of arguments for command download")
    else:
        id = args[0]
        downloader = EnvDownloader(id)
        if (len(args) == 1):
            downloader.download()
        else:
            downloader.download(args[1])

def do_config(args):
    gpt = GPT()
    log.info("################################################################################")
    print_config()
    log.info("The model has "+str(gpt.get_num_params())+" parameters")
    log.info("################################################################################")

def do_profile(args):
    if (args == None or len(args) < 2):
        log.error("Wrong number of arguments for command download")
    else:
        name = args[0]
        iterations = int(args[1]) 
        profile_run(name, iterations)

def do_tokenstats(args):
    stats = TokenStats()
    max_tokens = -1
    if (args != None):
        max_tokens = int(args[0])
    stats.generate(max_tokens)
    stats.print()

def do_ngrams(args):
    if (args == None):
        log.error("Wrong number of arguments for command ngrams")
    else:
        ngrams = Ngrams(readonly=False, index = int(args[0]))
        ngrams.generate()
        ngrams.close()

def do_ngrams_stats(args):
    ngrams = Ngrams(readonly=True)
    ngrams.print_stats()
    ngrams.close()

def do_startindex(args):
    tokenizer = Tokenizer()
    tokenizer.load_vocab()
    index = StartIndex(readonly=False)
    index.generate(tokenizer.vocab_map['<end/>'])

def do_dispress(args):
    generator = DisPressGenerator()
    generator.generate()

def do_score(args):
    if (args == None):
        log.error("Wrong number of arguments for command score")
    else:
        score = Score()
        if (len(args)>1):
            if (args[0] == '-file'):
                log.info("Score = "+str(score.calculate_from_file(args[1])))
            else:
                log.info("Score = "+str(score.calculate(args[0])))
        else:
            log.info("Score = "+str(score.calculate(args[0]))) 
        
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
elif (command == 'download'):
    do_download(args)
elif (command == 'config'):
    do_config(args)
elif (command == 'profile'):
    do_profile(args)
elif (command == 'tokenstats'):
    do_tokenstats(args)
elif (command == 'ngrams'):
    do_ngrams(args)
elif (command == 'ngrams-stats'):
    do_ngrams_stats(args)
elif (command == 'dispress'):
    do_dispress(args)
elif (command == 'startindex'):
    do_startindex(args)
elif (command == 'score'):
    do_score(args)
else:
    raise Exception('Unknown command: '+command)



