import multiprocessing
import sys
import os
import time
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

def worker(func, expID, args, kwargs):
   name = multiprocessing.current_process().name
   ret = func(expID, name, *args, **kwargs)
   if ret != 0 :
     print('%s has failed'%name)
   else:
     print('%s has succeeded'%name)


if __name__ == '__main__':
  nameList = ['u-11100000', 'u-11100600', 'u-11101200', 'u-11101800', 
      'u-11110000', 'u-11111200']
  func, args, kwargs = get_func(sys)
  jobs = []
  for expID in nameList:
    p  = multiprocessing.Process(target=worker, args=(func, expID, args, kwargs))
    jobs.append(p)
    p.start()


