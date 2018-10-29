from multiprocessing import Process, Value, current_process
import sys
import os
import time

import Worker

def get_func(ipt):
    exit='''
Usage:
    %s func_name [args] [kwargs]
    '''%sys.argv[0]
    try:
        funcn = sys.argv[1]
    except IndexError:
        sys.exit(exit)
    #possibles = globals().copy()
    #possibles.update(locals())
    #method = possibles.get(funcn)
    try:
        method = getattr(Worker, funcn)
    except AttributeError:
        raise NotImplementedError("Method %s not implemented" % funcn)
    args = []
    kwargs = {}
    try:
        for arg in sys.argv[2:]:
            try:
                key, value = arg.split('=')
                key = key.replace('-','')
                kwargs[key] = value
            except ValueError:
               args.append(arg)
    except IndexError:
        pass

    return method, args, kwargs

def worker(func, expID, exitValue, args, kwargs):
   name = current_process().name
   exitValue.value += 1
   ret = func(expID, name, *args, **kwargs)
   if ret != 0 :
     sys.stdout.write('%s has failed\n'%name)
   else:
     sys.stdout.write('%s has succeeded\n'%name)
     exitValue.value -= 1

def process_worker(nameList, func, args, kwargs):
  '''
   Distributes the jobs from nameList to each process
  '''
  jobs = []
  exitValue = Value('i',0)
  for expID in nameList:
    p  = Process(target=worker, args=(func, expID, exitValue, args, kwargs))
    jobs.append(p)
    p.start()

  while True:
    active = 0
    for job in jobs:
      active += int(job.is_alive())
    if not active:
      break
    time.sleep(1)
  if exitValue.value != 0:
    sys.stdout.write('%s had %i failed processes\n'\
                      %(os.path.basename(sys.argv[0]), exitValue.value))
  else:
    sys.stdout.write('%s has finished\n'\
                      %(os.path.basename(sys.argv[0])))
  sys.exit(exitValue.value)

    

if __name__ == '__main__':

  nameList = ['u-11091200',  'u-11091800',  'u-11100000',  'u-11100600',
              'u-11101200',  'u-11101800',  'u-11110000',  'u-11111200']
  #nameList = ('u-11100000', 'u-11101200')
  #nameList = ('u-11091200',)
  process_worker(nameList, *get_func(sys))
