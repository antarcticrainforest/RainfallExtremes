from netCDF4 import Dataset as nc, num2date
import pandas as pd
import numpy as np
import os,sys,gc
from matplotlib import pyplot as plt

def get_percentiles(fname, perc=list(range(1,100)), varname='rain_rate',
    out=os.path.join(os.getenv('HOME'),'Data'),group='6h'):
    '''
    Calculate percentile in a netcdf-file
    '''
    out=os.path.join(out,'Percentiles.hdf5')
    with nc(fname) as fn:
        name = 'P%s'%group
        group = fn.groups[group]
        present = group.variables['ispresent'][:]
        data = group.variables['rain_rate'][:]
        if group == '10min':
            data = data[present == 1]
        else:
            data = data[present> 75,...]
        data = data[data>0.1]
        percvals = np.percentile(data,perc)

        if not os.path.isfile(out):
            with pd.HDFStore(out,'w') as h5:
                h5['/perc'] = pd.DataFrame(percvals,index=perc, columns=[name])
        else:
            with pd.HDFStore(out,'a') as h5:
                DF = h5['/perc']
                S = pd.Series(percvals,index=perc)
                DF[name] = S
                h5['/perc'] = DF
        del data, percvals, present
        gc.collect()


if __name__ == '__main__':
    filename = os.path.join(os.getenv('HOME'),'Data','Darwin','netcdf','CPOL.nc')
    percs = list(range(0,100))+[99.9,99.99,99.999,100]
