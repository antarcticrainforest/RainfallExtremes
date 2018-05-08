from netCDF4 import Dataset as nc, num2date
import pandas as pd
import numpy as np
import os,sys,gc
from matplotlib import pyplot as plt

def get_percentiles(fname, perc=list(range(1,100)), varname='rain_rate', thresh=75,
    out=os.path.join(os.getenv('HOME'),'Data','Extremes','CPOL'),resol='km2.5',):
    '''
    Calculate percentile in a netcdf-file
    '''
    out = os.path.join(out,'CPOL_Percentiles.hdf5')
    groups = ('10min','1h','3h','6h','24h')
    with nc(fname) as fn:
      if not os.path.isfile(out):
        mode = 'w'
      else:
        mode = 'a'
        DF=pd.DataFrame(np.empty([len(perc),len(groups)]), columns=groups, index=perc)
      with pd.HDFStore(out, mode) as h5:
        for gr in groups:
            sys.stdout.flush()
            sys.stdout.write('\r Adding %s to %s' %(gr, os.path.basename(out)))
            sys.stdout.flush()
            name = 'P%s'%gr
            group = fn.groups[gr]
            if group == '10min':
                present = group.variables['ispresent'][:] == 1
            else:
                present = group.variables['ispresent'][:] > thresh
            data = np.ma.masked_outside(group.variables['rain_rate'][perc,:],0.1,100000)
            DF[group]= np.nanpercentile(data.filled(np.nan),perc)
            del data, present
            gc.collect()

        h5[resol] = DF


if __name__ == '__main__':
    filename = os.path.join(os.getenv('HOME'),'Data','Extremes','CPOL','CPOL_1998-2017.nc')
    percs = list(range(0,100))+[99.9,99.99,99.999,99.9999,100]
    resol = 'km2.5'
    get_percentiles(filename, perc=percs, resol=resol)
