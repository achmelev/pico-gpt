from environment import initEnv
from sys import argv

if (len(argv) < 3):
    raise Exception('Called with less as 2 arguments')

initEnv(argv[1])
from environment import log

command = argv[2]
args = None
if (len(argv) > 3):
    args = argv[3:]

def do_init(args):
    log.debug('do_init')

if (command == 'init'):
    do_init(args)


