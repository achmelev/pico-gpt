from sqlite3 import connect
from environment import log, workDir, get_int_config_value
from os import mkdir
from os.path import isfile, isdir 
from numpy import memmap, uint16
from progress import Progress


class Ngrams:
    def __init__(self, readonly = True):
        self.ngram_size = get_int_config_value('ngram_size')
        self.insert_interval = get_int_config_value('ngram_insert_interval')
        self.hashtable_size = get_int_config_value('ngram_hastable_size')

        self.readonly = readonly
        if readonly:
            self.active = isdir(workDir+"ngrams")
        else:
            assert not isdir(workDir+"ngrams"), workDir+"ngrams"+" already exists!"
            self.train_data = memmap(workDir+'train.bin', dtype=uint16, mode='r')
        if readonly and self.active:
            self.openReadConnections()
    
    def openReadConnections(self):
        self.connection = []
        self.cursor = []
        for idx in range(self.hashtable_size):
            self.connection.append(connect(workDir+"ngrams/data"+str(idx)+".db"))
            self.cursor.append(self.connection[idx].cursor())
    
    def openWriteConnections(self):
        assert not self.readonly,'opened in read mode'
        mkdir(workDir+"ngrams")
        self.connection = []
        self.cursor = []
        for idx in range(self.hashtable_size):
            self.connection.append(connect(workDir+"ngrams/data"+str(idx)+".db", isolation_level= None))
            cur = self.connection[idx].cursor()
            self.cursor.append(cur)
            #Pragmas (connection and db level)
            cur.execute('PRAGMA journal_mode = off')
            cur.execute('PRAGMA synchronous = 0')
            cur.execute('PRAGMA locking_mode = EXCLUSIVE')
            cur.execute('PRAGMA temp_store = MEMORY')
    
    def initdbs(self):
        assert not self.readonly,'opened in read mode'
        for cur in self.cursor:
            cur.execute('CREATE TABLE ngrams(ngram BLOB, next INTEGER)')
            cur.execute('CREATE INDEX ngrams_idx on ngrams(ngram)')
    
    def initWriteCache(self):
        self.writeCache = []
        for idx in range(self.hashtable_size):
            self.writeCache.append([])

    def fnv1a_64(self, ngram):
        #Constants
        FNV_prime = 16777619
        offset_basis = 2166136261

        #FNV-1a Hash Function
        hash = offset_basis
        for idx in range(len(ngram)):
            hash = hash ^ ngram[idx]
            hash = hash * FNV_prime
        return hash
    
    def getNgramsShardIndex(self, ngram):
        hashValue = self.fnv1a_64(ngram)
        idx = hashValue%self.hashtable_size
        return idx

    def flushWriteCache(self, idx):
        cur = self.cursor[idx]
        cur.executemany('INSERT INTO ngrams VALUES (?,?)',self.writeCache[idx])
        self.writeCache[idx] = []

    def flushWriteCaches(self):
        for idx in range(len(self.writeCache)):
            if len(self.writeCache[idx]) >0:
                self.flushWriteCache(idx)

    def writeChunk(self, chunk):
        ngram = chunk[:-1].tobytes()
        nextToken = int(chunk[len(chunk)-1])
        idx = self.getNgramsShardIndex(ngram)
        self.writeCache[idx].append([ngram, nextToken])
        if (len(self.writeCache[idx]) >= self.insert_interval):
            self.flushWriteCache(idx)

    def generate(self):
        log.info("Generating ngrams db...")
        self.openWriteConnections()
        self.initdbs()
        self.initWriteCache()

        chunk_size = self.ngram_size+1

        log.info("Scanning train data...")
        progress = Progress(len(self.train_data)-chunk_size-1, 100)
        for idx in range(len(self.train_data)-chunk_size):
            chunk = self.train_data[idx:idx+chunk_size]
            self.writeChunk(chunk)
            progress.update(idx)
        log.info("Done")
        log.info("Flushing cache")
        self.flushWriteCaches()
        log.info("Done")
    
    def print_stats(self):
        assert self.readonly,'opened in write mode'
        assert self.active,'ngrams not active'
        print("############################################")
        print('Got '+str(self.hashtable_size)+" shards")
        for idx in range(self.hashtable_size):
            cur = self.cursor[idx]
            cur.execute('SELECT count(ngram) from ngrams')
            result = cur.fetchone()[0]
            print(str(result)+" entries in the shard "+str(idx))
        print("############################################")
        

    
    def get_ngram_nexts(self, ngram):
        assert len(ngram) == self.ngram_size,'Wrong ngram size: '+str(len(ngram))
        assert self.readonly,'opened in write mode'
        assert self.active,'ngrams not active'
        ngram_bytes = ngram.tobytes()
        cur = self.cursor[self.getNgramsShardIndex(ngram_bytes)]
        values = [ngram_bytes]
        cur.execute('SELECT next from ngrams where ngram = ?',values)
        sqlresult =  cur.fetchall()
        cur.close()
        return [x[0] for x in sqlresult]
    
    def close(self):
        for cur in self.cursor:
            cur.close()
        for con in self.connection:
            con.close()
        


