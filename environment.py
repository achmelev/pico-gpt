from configparser import ConfigParser
from logging import Logger, StreamHandler, Formatter, getLogger, WARN, INFO, DEBUG, ERROR, FATAL
from os.path import isfile, isdir
from os import mkdir
from sys import stdout
import torch

environmentName : str = None
workDir: str = None

config: ConfigParser = None
log: Logger = None

device: str = None

def checkInitialized():
    global environmentName
    if (environmentName == None):
        raise Exception('Environment not initialized!')

def get_config_value(name: str) -> str:
    global config
    checkInitialized()
    return config.get(environmentName, name)

def get_int_config_value(name: str) -> int:
    global config
    checkInitialized()
    return config.getint(environmentName, name)

def get_bool_config_value(name: str) -> bool:
    global config
    checkInitialized()
    return config.getboolean(environmentName, name)

def get_float_config_value(name: str) -> float:
    global config
    checkInitialized()
    return config.getfloat(environmentName, name)

def get_env_filename(filename: str) -> str:
    global workDir
    return workDir+filename

def initEnv(name: str):
    global environmentName
    global workDir
    global log
    global config
    global device

    #environmentName
    environmentName = name
    #Config
    if (config == None):
        config = ConfigParser()
        config.read('pico_gpt.conf')
    if (not config.has_section(environmentName)):
        raise Exception('Unknown environment name: '+environmentName)
    #Logging
    log = getLogger('pico_gpt')
    if (len(log.handlers) == 0):
        ch = StreamHandler( stream=stdout)
        loglevelStr = get_config_value('loglevel')
        if (loglevelStr.lower() == 'debug'):
            log.setLevel(DEBUG)
        elif (loglevelStr.lower() == 'info'):
            log.setLevel(INFO)
        elif (loglevelStr.lower() == 'warn'):
            log.setLevel(WARN)
        elif (loglevelStr.lower() == 'error'):
            log.setLevel(ERROR)
        elif (loglevelStr.lower() == 'fatal'):
            log.setLevel(FATAL)
        else:
            raise Exception('Unknown log level: '+loglevelStr)
        formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(formatter)
        log.addHandler(ch)
    #Work Directory
    workDir = './work_'+environmentName+'/'
    if (isfile(workDir)):
        raise Exception (workDir+' exists and is a file!')
    elif (isdir(workDir)):
        pass
    else:
        log.info('Creating environment work directory '+workDir)
        mkdir(workDir)

    #Device
    if torch.cuda.is_available():
        log.info("CUDA is available")
        if config.getboolean('use_cuda'):
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        log.info("CUDA isn't available")
        device = 'cpu'
    log.info("Using device "+device)

    log.info('Enviroment '+environmentName+' initialized.')







