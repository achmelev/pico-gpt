class Progress:
    def __init__(self, max, progress_length):
        self.progress = 0
        self.max = max
        self.progress_length = progress_length
        self.done = False
    
    def update(self,idx):
        if self.done:
            return
        if (idx >= self.max):
            print("")
            self.done = True
            return
        new_progress = int(float(self.progress_length)*float(idx)/float(self.max))
        if (new_progress > self.progress):
            self.progress = new_progress
            print("Done "+str(self.progress)+" of "+str(self.progress_length))

