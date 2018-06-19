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
        res = args[0]
    except IndexError:
        res = '0p44km'
    fileID = dict(ph='rain', pg='geop_th', pf='pres_th', pe='qcl_th',
                  pd='qcf_th', pc='vert_cent', pb='vert_wind', pa='surf',
                  pvera='vera', pverb='verb', pverc='verc')
    expID_tmp = expID.replace('u-','u-2006')
    date = datetime.strptime(expID_tmp, 'u-%Y%m%d%H%M')
    umID = date.strftime('%Y%m%dT%H%MZ')

    old_path = os.path.dirname(__file__)
    path = os.path.join(os.getenv('HOME'), 'cylc-run', expID, 'share', 'cycle',
                        umID, res, 'protoRA1T', 'um')
    if not os.path.isdir(path):
        print('%s : path does not exist')
        return 1

    os.chdir(path)
    for umid, ncid in fileID.items():
        for umfile in glob('umnsaa_%s*'%umid):
            os.system('echo um2cdf %s'%umfile)
            num = int(umfile.split('_')[-1].replace(umid,''))
            outdate = (date + timedelta(hours=num)).strftime('%Y%m%d_%H%M')
            outfile = 'um-%s-%s_%s.nc'%(res, expID.replace('u-',''), outdate)
            os.system('echo mv %s.nc %s' %(umfile, outfile))
    return 0



