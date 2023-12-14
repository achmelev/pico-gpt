from requests import Session
import tarfile
from os import remove

from environment import log, workDir


class EnvDownloader:

    def __init__(self, id):
        self.id = id
    
    def download(self, tofile = None):
        if (tofile == None):
            downloadFile = workDir+"temp.tgz"
        else:
            downloadFile = workDir+tofile
        log.info("Downloading file from Google Drive with id "+self.id+" to "+downloadFile)
        url = "https://drive.usercontent.google.com/download"
        payload = {'id': self.id, 'export': 'download', 'authuser':'0','confirm':'t'}
        download = open(downloadFile,"wb")
        session  = Session()
        response = session.get(url, params = payload, stream=True)
        if (response.status_code == 200):
            log.info("Writing file...")
            bytes_counter = 0
            for chunk in response.iter_content(chunk_size=10*1024*1024):
                download.write(chunk)
                bytes_counter+=len(chunk)
                log.info("Got "+str(bytes_counter)+" bytes")
            download.close()
            log.info("Done")
            if (tofile == None):
                tar = tarfile.open(downloadFile)
                log.info("Extracting environment files...")
                tar.extractall(path=workDir)
                log.info("Done")
                remove(downloadFile)
        else:
            log.error("Couldn't download, statusCode = "+str(response.status_code)+", reason = "+response.reason)

        
