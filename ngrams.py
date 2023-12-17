from sqlite3 import connect
from environment import log, workDir, get_int_config_value
from os.path import isfile, isdir 
from numpy import memmap, uint16

class Ngrams:
    def __init__(self, readonly = True):
        self.ngram_size = get_int_config_value('ngram_size')
        self.commit_interval = get_int_config_value('ngram_commit_interval')
        self.readonly = readonly
        if readonly:
            self.active = isfile(workDir+"ngrams.db")
        else:
            assert not isfile(workDir+"ngrams.db"), workDir+"ngrams.db"+" already exists!"
            self.active = True
            self.initialized = False
            self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
        if (self.active):
            self.connection = connect(workDir+"ngrams.db")
    
    def initdb(self):
        assert not self.readonly,'opened in read mode'
        assert not self.initialized, 'already initialized'

        cur = self.connection.cursor()
        cur.execute('CREATE TABLE ngrams(ngram BLOB, idx INTEGER)')
        cur.execute('CREATE INDEX ngrams_idx on ngrams(ngram)')
        cur.execute('CREATE TABLE start_pos(idx INTEGER, pos INTEGER, PRIMARY KEY(idx))')
        cur.close()
    
    def generate(self, start_token = 0):
        log.info("Generating ngrams db...")
        cur = self.connection.cursor()
        start_pos_idx = 0
        for idx in range(len(self.train_data)-self.ngram_size):
            chunk = self.train_data[idx:idx+self.ngram_size]
            values = [chunk.tobytes(),idx]
            cur.execute('INSERT INTO ngrams VALUES (?,?)',values)
            if (chunk[0] == start_token):
                values = [start_pos_idx, idx]
                cur.execute('INSERT INTO start_pos VALUES (?,?)',values)
                start_pos_idx+=1
            if (idx+1)%self.commit_interval == 0:
                log.debug('Written '+str(idx+1)+" ngrams")
                self.connection.commit()
        cur.close()
        self.connection.commit()
        log.info("Done")
    
    def count_ngram(self, ngram):
        assert len(ngram) == self.ngram_size,'Wrong ngram size: '+str(len(ngram))
        assert self.readonly,'opened in write mode'
        assert self.active,'ngrams not active'
        cur = self.connection.cursor()
        values = [ngram.tobytes()]
        cur.execute('SELECT count(ngram) from ngrams where ngram = ?',values)
        result =  cur.fetchone()[0]
        cur.close()
        return result
    
    def count_start_pos(self):
        assert self.readonly,'opened in write mode'
        assert self.active,'ngrams not active'
        cur = self.connection.cursor()
        cur.execute('SELECT count(idx) from start_pos')
        result =  cur.fetchone()[0]
        cur.close()
        return result
    
    def get_start_pos(self, idx):
        assert self.readonly,'opened in write mode'
        assert self.active,'ngrams not active'
        cur = self.connection.cursor()
        cur.execute('SELECT pos from start_pos where idx = ?', [idx])
        result =  cur.fetchone()[0]
        cur.close()
        return result

    def close(self):
        self.connection.close()
        


