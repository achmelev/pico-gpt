
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