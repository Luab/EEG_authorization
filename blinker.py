from psychopy import logging, core

import datetime
import timeit
from multiprocessing import Process, freeze_support, RLock

import open_bci_v3 as bci


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

clock = core.Clock()
now = datetime.datetime.now()
logging.LogFile(f='%d-%d-%d_%d-%d-%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second),level=logging.DATA)
logging.setDefaultClock(clock)

class CSVLogger():
    def __init__(self, file_name="collect.csv", delim=",", verbose=False):

        now = datetime.datetime.now()
        self.time_stamp = '%d-%d-%d_%d-%d-%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        self.file_name = 'test'
        self.start_time = timeit.default_timer()
        self.delim = delim
        self.verbose = verbose


    def logger(self, sample):
        t = timeit.default_timer() - self.start_time
        # print(timeSinceStart|Sample Id)
       # print("CSV: %f | %d" % (t, sample.id))
        row = ''
        row += str(t)
        row += self.delim
        row += str(sample.id)
        row += self.delim
        for i in sample.channel_data:
            row += str(i)
            row += self.delim
        for i in sample.aux_data:
            row += str(i)
            row += self.delim
        # remove last comma
        row += '\n'
        with open(self.file_name, 'a') as f:
            f.write(row)


def blinking(lock,i):
    lock.acquire()
    start = timeit.default_timer()
    from psychopy import visual
    win = visual.Window([1920,1080],allowGUI=False,fullscr=True,color=(255,255,255))
    message = visual.Circle(win,lineColor='green',fillColor='green')
    message.setAutoDraw(True)  # automatically draw every frame
    win.autoLog = True
    lock.release()
    for i in range(100):
        win.flip()
        core.wait(0.09)
        message.opacity = 0  # change properties of existing stim
        with open('test', 'a') as f:
            f.write(str(timeit.default_timer() - start)+",'blink'"+"\n")
        win.flip()
        core.wait(0.09)
        message.opacity = 1

def board(lock,i):
    lock.acquire()
    Logger = CSVLogger()
    board = bci.OpenBCIBoard(port="COM3",
                         filter_data=False,
                         scaled_output=True,
                         log=True,
                         aux=False)

    import threading
    print(board.streaming)
    boardThread = threading.Thread(target=board.start_streaming, args=(Logger.logger, -1))
    boardThread.start()
    print(board.streaming)
    x = True
    while x:
        if board.streaming:
            lock.release()
            x = False



if __name__ == '__main__':
    freeze_support()
    lock = RLock()

    boardProcess = Process(target=board,args=(lock,1))
    blinkProcess = Process(target=blinking,args=(lock,0))
    boardProcess.start()
    blinkProcess.start()
    boardProcess.join()
    blinkProcess.join()