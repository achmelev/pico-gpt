from os.path import isfile, getsize
from mmap import mmap, PROT_WRITE, PROT_READ


class TokenTreeNode:
    
    def  __init__(self, content = None):
        if (content == None):
            self.content = bytearray(13)
        else:
            assert len(content) == 13, 'Wrong error size: '+str(len(content))
            self.content = content
    
    def get_token(self):
        return int.from_bytes(self.content[0:2], byteorder='big')
    
    def set_token(self, value):
        self.content[0:2] = value.to_bytes(2, byteorder='big')
    
    def get_count(self):
        return int.from_bytes(self.content[2:5], byteorder='big')
    
    def set_count(self, value):
        self.content[2:5] = value.to_bytes(3, byteorder='big')
    
    def get_sibling(self):
        return int.from_bytes(self.content[5:9], byteorder='big')
    
    def set_sibling(self, value):
        self.content[5:9] = value.to_bytes(4, byteorder='big')
    
    def get_child(self):
        return int.from_bytes(self.content[9:13], byteorder='big')
    
    def set_child(self, value):
        self.content[9:13] = value.to_bytes(4, byteorder='big')
    
    token = property (
        fget=get_token,
        fset=set_token
    )

    count = property (
        fget=get_count,
        fset=set_count
    )

    sibling = property (
        fget=get_sibling,
        fset=set_sibling
    )

    child = property (
        fget=get_child,
        fset=set_child
    )

class TokenTree:

    def __init__(self, file, mode):
        assert mode=='w' or mode=='r','Illegal mode '+mode
        if (mode == 'r'):
            assert isfile(file),'File '+str(file)+'does not exist!'
        self.file = file
        self.mode = mode

        if (isfile(self.file)):
            assert getsize(self.file) >0 and getsize(self.file)%4096 == 0,'Wrong file size: '+str(getsize(self.file))
            self.pageSize = getsize(self.file)/4096
            if (self.mode == 'r'):
                self.f = open(self.file,'rb')
                self.mm = mmap(self.f.fileno(), 0, prot=PROT_READ)
            else:
                self.f = open(self.file,'r+b')
                self.mm = mmap(self.f.fileno(), 0, prot=PROT_WRITE)
            
        else:
            self.pageSize = 1
            self.f = open(self.file,'wb')
            self.f.write(bytearray(4096))
            self.f.close()
            self.f = open(self.file,'r+b')
            self.mm = mmap(self.f.fileno(), 4096, prot=PROT_WRITE)

    def appendPage(self):
        assert self.mode == 'w','Read-Only tree'
        self.pageSize+=1
        self.mm.resize(self.pageSize*4096)

    def close(self):
        self.mm.flush()
        self.mm.close()
        self.f.close()


        



