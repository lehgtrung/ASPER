
from datetime import datetime


class Logger:
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        with open(self.path, 'a') as f:
            f.write(f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}\n')
            f.write(msg + '\n\n')

    def raw_info(self, msg):
        with open(self.path, 'a') as f:
            f.write(msg + '\n')
