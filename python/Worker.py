'''
Python module with functions that does the Ensemble manipulation
'''
from glob import glob
from datetime import datetime, timedelta
import os, sys
import time, random


def test(expID, thread):
  '''
  Make some tests for the multiprocessing environment
  '''

  print('%s is working on %s'%(thread, expID))
  time.sleep(4*random.random())
  return 0 #random.randint(0,1)

def um2nc(expID, thread, *args, **kwargs):
    '''
    Convert all um output to netcdf-file format and rename the files 
    to a meaningful name
    '''
    try:
        res = args[1]
        setup='proto_RA1T'
    except IndexError:
        res = '0p44km'
        setup='protoRA1T'

    fileID = dict(ph='rain', pg='geop_th', pf='pres_th', pe='qcl_th',
                  pd='qcf_th', pc='vert_cent', pb='vert_wind', pa='surf',
                  pvera='vera', pverb='verb', pverc='verc')
    try:
      fileID = {args[0]:fileID[args[0]]}
    except IndexError:
      fileID = fileID

    expID_tmp = expID.replace('u-','u-2006')
    date = datetime.strptime(expID_tmp, 'u-%Y%m%d%H%M')
    umID = date.strftime('%Y%m%dT%H%MZ')

    old_path = os.path.dirname(__file__)
    path = os.path.join(os.getenv('HOME'), 'cylc-run', expID, 'share', 'cycle',
                        umID, 'darwin', res, setup, 'um')
    outpath = os.path.join(os.getenv('HOME'), 'Data', 'UM_1', 'RA1T', umID,
                           'darwin', res)
    if not os.path.isdir(outpath):
      if os.system('mkdir -p %s'%outpath) != 0:
        print('Making dir %s failed'%outpath)
        return 1
    if not os.path.isdir(path):
        print('%s : path does not exist'%path)
        return 1
    os.chdir(outpath)
    def exec(thread, cmd):
      '''Execute command an leave'''
      sys.stdout.write('%s: Executing %s \n'%(thread, cmd))
      out = os.system(cmd)
      if out != 0:
        sys.stdout.write('%s: Error executing %s \n'%(thread, cmd))

      return out

    for umid, ncid in fileID.items():
        outdates = []
        umfiles = glob(os.path.join(path,'umnsaa_%s*'%umid))
        umfiles.sort()
        if len(umfiles) == 0:
          sys.stdout.write('No files are excisting in %s\n'%path)
          return 1
        merge = True
        for umfile in umfiles:
            testfile = glob('um-%s-%s-%s_????????_????-????????_????.nc'\
            %(res, expID.replace('u-',''), ncid))
            if not len(testfile):
                merge = True
                cmd = 'cp %s .'%umfile
                if exec(thread, cmd) != 0:
                  return 1
                cmd = 'um2cdf %s'%(os.path.basename(umfile))
                exec(thread, cmd)
                cmd = 'rm %s'%(os.path.basename(umfile))
                exec(thread, cmd)
                num = int(umfile.split('_')[-1].replace(umid,''))
                outdate = (date + timedelta(hours=num)).strftime('%Y%m%d_%H%M')
                outfile = 'um-%s-%s-%s_%s.nc'%(res, expID.replace('u-',''), ncid,
                           outdate)
                cmd2 = 'cdo sellonlatbox,130.024,131.58,-11.99,-11.083 %s.nc %s' %(os.path.basename(umfile), outfile)
                outdates.append(outdate)
                if exec(thread, cmd2) != 0:
                  return 1
                cmd2 = 'rm %s.nc'%(os.path.basename(umfile))
                if exec(thread, cmd2) != 0:
                  return 1
            else:
                merge = False

        outdates.sort()
        if merge:
            mergefile = 'um-%s-%s-%s_%s-%s.nc'%(res, expID.replace('u-',''), ncid,
                                                outdates[0],outdates[-1])
            cdofiles = 'um-%s-%s-%s_'%(res, expID.replace('u-',''), ncid)
            cmd3 = 'cdo mergetime %s* %s'%(cdofiles, mergefile)
            if exec(thread, cmd3) != 0:
              return 1

            cmd4 = 'rm %s????????_????.nc'%cdofiles
            if exec(thread, cmd4) != 0:
              return 1
    os.chdir(old_path)
    return 0



