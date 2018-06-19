'''
Python module with functions that does the Ensemble manipulation
'''
from glob import glob
from datetime import datetime, timedelta
import os

def um2nc(expID, thread, *args, **kwargs):
    '''
    Convert all um output to netcdf-file format and rename the files 
    to a meaningful name
    '''
    try:
        res = args[1]
    except IndexError:
        res = '0p44km'

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
                        umID, 'darwin', res, 'protoRA1T', 'um')
    if not os.path.isdir(path):
        print('%s : path does not exist'%path)
        return 1
    os.chdir(path)
    for umid, ncid in fileID.items():
        outdates = []
        umfiles = glob('umnsaa_%s*'%umid)
        umfiles.sort()
        merge = True
        for umfile in umfiles:
            testfile = glob('um-%s-%s-%s_????????_????-????????_????.nc'\
            %(res, expID.replace('u-',''), ncid))
            if not len(testfile):
                merge = True
                cmd='um2cdf %s'%umfile
                print('%s: %s'%(thread, cmd))
                ret = os.system(cmd)
                if ret != 0:
                  print('%s: Error moving'%thread)
                  os.system("echo %s $PWD"%thread)
                  return 1
                num = int(umfile.split('_')[-1].replace(umid,''))
                outdate = (date + timedelta(hours=num)).strftime('%Y%m%d_%H%M')
                outfile = 'um-%s-%s-%s_%s.nc'%(res, expID.replace('u-',''), ncid,
                           outdate)
                cmd2 = 'mv %s.nc %s' %(umfile, outfile)
                outdates.append(outdate)
                print('%s: %s'%(thread, cmd2))
                ret = os.system(cmd2)
                if ret != 0:
                  print('%s: Error moving'%thread)
                  os.system("echo %s $PWD"%thread)
                  return 1
            else:
                merge = False

        outdates.sort()
        if merge:
            mergefile = 'um-%s-%s-%s_%s-%s.nc'%(res, expID.replace('u-',''), ncid,
                                                outdates[0],outdates[-1])
            cdofiles = 'um-%s-%s-%s_'%(res, expID.replace('u-',''), ncid)
            cmd3 = 'cdo mergetime %s* %s'%(cdofiles, mergefile)
            print('%s: %s'%(thread, cmd3))
            os.system(cmd3)
            cmd4 = 'rm %s????????_????.nc'%cdofiles
            print('%s: %s'%(thread, cmd4))
            os.system(cmd4)
    os.chdir(old_path)
    return 0



