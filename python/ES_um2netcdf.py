'''
Module to create netcdf-files for the ensemble run
'''

import queue as Queue
import threading
import numpy as np
import os
import sys
import time
import sys
global exitFlag
exitFlag = 0




class EsThread(threading.Thread):
    def __init__(self, threadID, name, q, lock, func, args, kwargs):
        super(EsThread, self).__init__()
        self.threadID = threadID
        self.name = name
        self.q = q
        self.lock = lock
        self.args = args
        self.kwargs = kwargs
    def run(self):
        print("Starting to convert: %s" %(self.name))
        worker(func, self.name, self.q, self.lock, self.args, self.kwargs)

#@synchronized
def worker(func, threadName, q, queueLock, args, kwargs):

    while not exitFlag:
        queueLock.acquire()
        if not q.empty():
            expID = q.get()
            queueLock.release()
            print("%s processing %s" % (threadName, expID))
            try:
                exitFlag += func(*args, **kwargs)
            except:
                exitFlag += 1
        else:
            queueLock.release()
        time.sleep(1)

def get_func(ipt):
    from Worker import um2nc
    exit='''
Usage:
    %s func_name [args] [kwargs]
    '''%sys.argv[0]
    try:
        funcn = sys.argv[1]
    except IndexError:
        sys.exit(exit)
    possibles = globals().copy()
    possibles.update(locals())
    method = possibles.get(funcn)
    if not method:
        raise NotImplementedError("Method %s not implemented" % funcn)
    args = []
    kwargs = {}
    try:
        for arg in sys.argv[2:]:
            try:
                key, value = arg.split('=')
                try:
                    kwargs[key] = int(value)
                except ValueError:
                    try:
                        kwargs[key] = float(value)
                    except ValueError:
                        kwargs[key] = value
            except ValueError:
                try:
                    args.append(int(arg))
                except ValueError:
                    try:
                        args.append(float(arg))
                    except ValueError:
                        args.append(arg)
    except IndexError:
        pass

    return method, args, kwargs


#if __name__ == '__main__':
    func, args, kwargs = get_func(sys)
    nameList = ['u-11100000', 'u-11100600', 'u-11101200', 'u-11101800',
                'u-11110000', 'u-11110600', 'u-11111200']
    threadList = ['Thread-%02i'%i for i in range(7)]
    workQueue = Queue.Queue(len(nameList))
    threads = []
    threadID = 1
    lock = threading.Lock()
    print(threadList)
    # Create new threads
    for tName in threadList:
        thread = EsThread(threadID, tName, workQueue, lock, func, args, kwargs)
        thread.start()
        threads.append(thread)
        threadID += 1

    # Fill the queue
    lock.acquire()
    for expID in nameList:
        workQueue.put(expID)
    lock.release()

    # Wait for queue to empty
    while not workQueue.empty():
        pass

    # Notify threads it's time to exit
    exitFlag = 1

    # Wait for all threads to complete
    for t in threads:
         t.join()
    print("Exiting Main Thread")







