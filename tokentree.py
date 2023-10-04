from os.path import isfile, getsize
from mmap import mmap, PROT_WRITE, PROT_READ


class TokenTreeNode:
    
    def  __init__(self, content:bytearray = None):
        self.index = None
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
        self.vocab_size = 0

        if (isfile(self.file)):
            assert getsize(self.file) >0 and getsize(self.file)%4096 == 0,'Wrong file size: '+str(getsize(self.file))
            self.pageSize = getsize(self.file)/4096
            if (self.mode == 'r'):
                self.f = open(self.file,'rb')
                self.mm = mmap(self.f.fileno(), 0, prot=PROT_READ)
            else:
                self.f = open(self.file,'r+b')
                self.mm = mmap(self.f.fileno(), 0, prot=PROT_WRITE)
            self.readHeader()
            assert (8+13*self.size) <=self.pageSize*4096,"Too small disk size: "+str(self.pageSize*4096)
            
        else:
            self.pageSize = 1
            self.f = open(self.file,'wb')
            self.f.write(bytearray(4096))
            self.f.close()
            self.f = open(self.file,'r+b')
            self.mm = mmap(self.f.fileno(), 4096, prot=PROT_WRITE)
            self.size = 0
            self.depth = 0
        
        
    
    def readHeader(self):
        self.size = int.from_bytes(self.mm[:4],'big')
        self.depth = int.from_bytes(self.mm[4:6],'big')
        self.vocab_size = int.from_bytes(self.mm[6:8],'big')

    def writeHeader(self):
        assert self.mode=='w',"Read only"
        self.mm[:4] = self.size.to_bytes(4,'big')
        self.mm[4:6] = self.depth.to_bytes(2, 'big')
        self.mm[6:8] = self.vocab_size.to_bytes(2, 'big')
    
    def appendNode(self, node):
        assert self.mode=='w',"Read only"
        assert node.index == None, 'Appending existing node!'
        offset = 8+13*self.size
        if (offset+13 > self.pageSize*4096):
            self.appendPage()
        self.mm[offset:offset+13] = node.content
        node.index = self.size
        self.size+=1

    
    def readNode(self, index):
        result = None
        if (result != None):
            return result
        assert 8+13*(index+1) <= self.pageSize*4096,'Out of bounds '+str(8+13*(index+1))+":"+str(self.pageSize*4096)
        result =  TokenTreeNode(bytearray(self.mm[8+13*index:8+13*(index+1)]))
        result.index = index
        return result
    
    def getLevel1Node(self, token):
        return self.readNode(token)
    
    def writeNode(self, node):
        assert node.index != None, 'Index not set'
        index = node.index
        assert self.mode=='w',"Read only"
        assert self.size>index,'Index out of bounds: '+str(index)+":"+str(self.size)
        assert 8+13*(index+1) <= self.pageSize*4096,'Out of bounds '+str(8+13*(index+1))+":"+str(self.pageSize*4096)

        self.mm[8+13*index:8+13*(index+1)] = node.content
    
    def searchTokenNode(self, parentNode, token):
        if (parentNode.child == 0):
            return None, None
        else:
            currentNode = self.readNode(parentNode.child)
            while (currentNode.token != token and currentNode.sibling != 0):
                currentNode = self.readNode(currentNode.sibling)
            if (currentNode.token == token):
                return currentNode, None
            else:
                return None, currentNode

    def verifyTokenPath(self, tokenPath):
        assert len(tokenPath) > 0, "Empty token path"
        for token in tokenPath:
            assert token < self.vocab_size

    def initFirstLevel(self, vocab_size):
        assert self.mode=='w',"Read only"
        assert self.size == 0,"Not empty tree"
        self.vocab_size = vocab_size
        for token in range(vocab_size):
            node = TokenTreeNode() 
            node.token = token
            node.child = 0
            node.sibling = 0
            node.count  = 0
            self.appendNode(node)
        self.depth = 1
    
    def createNewNode(self, token):
        node = TokenTreeNode()
        node.token = token
        node.count = 1
        node.child = 0
        node.sibling = 0
        return node
    
    def createFirstChild(self, parentNode, token):
        assert parentNode.child == 0,'something wrong, child index is already set'
        newNode = self.createNewNode(token)
        self.appendNode(newNode)
        parentNode.child = self.size-1
        self.writeNode(parentNode)

    def appendSibling(self, lastSiblingNode, token):
        assert lastSiblingNode.sibling == 0,'something wrong, child index is already set'
        newNode = self.createNewNode(token)
        self.appendNode(newNode)
        lastSiblingNode.sibling = self.size-1
        self.writeNode(lastSiblingNode)    

    
    def insertOrUpdateToken(self, tokenPath):
        assert self.depth >=1,"Empty tree"
        self.verifyTokenPath(tokenPath)
        assert len(tokenPath) <= self.depth+1, 'Token path too long, add prefixes first!'

        if len(tokenPath) == 1:
            node = self.getNode(tokenPath)
            node.count+=1
            self.writeNode(node)
        else:
            parentNode = self.getNode(tokenPath[:-1])
            assert parentNode != None, 'parent node missing, add prefixes first!'
            if (self.depth+1 == len(tokenPath)):
                self.createFirstChild(parentNode, tokenPath[-1:][0])
                self.depth+=1
            else:
                if (parentNode.child == 0):
                    self.createFirstChild(parentNode, tokenPath[-1:][0])
                else:
                    node, last_sibling = self.searchTokenNode(parentNode,tokenPath[-1:][0])
                    if (node != None):
                        node.count+=1
                        self.writeNode(node)
                    else:
                        assert last_sibling.sibling == 0,'sibling index set, something wrong'
                        self.appendSibling(last_sibling, tokenPath[-1:][0])


    def getNode(self, tokenPath):
        assert self.depth >=1,"Empty tree"
        self.verifyTokenPath(tokenPath)
        assert len(tokenPath) <= self.depth, 'Token path too long'
        for index, token in enumerate(tokenPath):
            assert token < self.vocab_size, 'token out of bounds!'
            if (index == 0):
                currentNode = self.getLevel1Node(token)
                if (len(tokenPath) == 1):
                    return currentNode
                if (currentNode == None):
                    return None
            elif (index > 0 and index < len(tokenPath)-1):
                currentNode, _ = self.searchTokenNode(currentNode, token)
                if (currentNode == None):
                    return None
            else:
                currentNode, _ = self.searchTokenNode(currentNode, token)
                return currentNode
    

    def getNodesChildrenList(self, parentNode):
        result = []
        if (parentNode.child == 0):
            return result
        else:
            currentNode = self.readNode(parentNode.child)
            result.append(currentNode)
            while (currentNode.sibling != 0):
                currentNode = self.readNode(currentNode.sibling)
                result.append(currentNode)
        return result


    def getNodesChildren(self, tokenPath):
        result = {}
        if (len(tokenPath) == 0):
            for token in range(self.vocab_size):
                result[token] = self.getLevel1Node(token)
        else:
            parentNode = self.getNode(tokenPath)
            
            if (parentNode == None):
                return result
            
            if (parentNode.child == 0):
                return result
            else:
                currentNode = self.readNode(parentNode.child)
                result[currentNode.token] = currentNode
                while (currentNode.sibling != 0):
                    currentNode = self.readNode(currentNode.sibling)
                    result[currentNode.token] = currentNode
        return result

    def appendPage(self):
        assert self.mode == 'w','Read-Only tree'
        self.pageSize+=1
        self.mm.resize(int(self.pageSize*4096))
    
    def doTraverse(self, nodesPath,callback):
        callback.onTraverse(nodesPath)
        children = self.getNodesChildrenList(nodesPath[len(nodesPath)-1])
        for childNode in children:
            nodesPath.append(childNode)
            self.doTraverse(nodesPath, callback)
            nodesPath.pop(len(nodesPath)-1)


    def traverse(self, callback):
        assert self.mode=='r','Traversing only an read-only tree'
        for token in range(self.vocab_size):
            node = self.getLevel1Node(token)
            if (node.count > 0):
                self.doTraverse([node], callback)
            


    def close(self):
        if (self.mode == 'w'):
            self.writeHeader()
            self.mm.flush()
        self.mm.close()
        self.f.close()


        



