from multiprocessing import Process, Value, current_process
import sys
import os
import time
def get_func(ipt):
    from Worker import um2nc, test
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

def worker(func, expID, exitValue, args, kwargs):
   name = current_process().name
   exitValue.value += 1
   ret = func(expID, name, *args, **kwargs)
   if ret != 0 :
     sys.stdout.write('%s has failed\n'%name)
   else:
     sys.stdout.write('%s has succeeded\n'%name)
     exitValue.value -= 1


if __name__ == '__main__':
  nameList = ['u-11100000', 'u-11100600', 'u-11101200', 'u-11101800', 
      'u-11110000', 'u-11111200']
  func, args, kwargs = get_func(sys)
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
  print(exitValue.value)
  if exitValue.value != 0:
    sys.stdout.write('%s had %i failed processes\n'\
                      %(os.path.basename(sys.argv[0]), exitValue.value))
  else:
    sys.stdout.write('%s has finished\n'\
                      %(os.path.basename(sys.argv[0])))
  sys.exit(exitValue.value)



