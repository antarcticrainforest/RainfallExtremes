from netCDF4 import Dataset as nc, num2date, date2num
import pandas as pd
import numpy as np
import os,sys,gc
from matplotlib import pyplot as plt

def get_percentiles(fname, perc=list(range(1,100)), varname='rain-rate', thresh=75,
    out=os.path.join(os.getenv('HOME'),'Data','Extremes','CPOL'),resol='all',):
    '''
    Calculate percentile in a netcdf-file
    '''
    out = os.path.join(out,'CPOL_TIWI_Percentiles.nc')
    groups = ('10min', '1h', '3h', '6h', '24h')
    grnum = np.array([10, 60, 3*60, 6*60, 24*60])
    burstf = os.path.join(os.getenv('HOME'), 'Data', 'Extremes', 'CPOL', 'CPOL_burst.pkl')
    burstdf = pd.read_pickle(burstf)
    if resol == 'burst':
      burst = burstdf['burst'].loc[burstdf['burst']==1].index
    elif resol == 'break':
      burst = burstdf['burst'].loc[burstdf['burst']==0].index
    else:
      burst = burstdf['burst'].dropna().index

    with nc(fname) as fn:
      if not os.path.isfile(out):
        mode = 'w'
      else:
        mode = 'a'
      DF=pd.DataFrame(np.empty([len(perc),len(groups)]), columns=groups, index=perc)
      with pd.HDFStore(out.replace('.nc','.hdf5'), mode) as hdf5:
        with nc(out, mode) as h5:
          try:
            h5.createGroup(resol)
          except (RuntimeError, ValueError, OSError):
            pass
          for key, value in (('perc', perc), ('groups', groups)):
            try:
              h5.createDimension(key, len(value))
            except (RuntimeError, OSError, ValueError):
              pass
            try:
              h5.createVariable(key, 'f', (key, ))
            except (RuntimeError, OSError, ValueError):
              pass
          try:
            h5.groups[resol].createVariable(varname, 'f', ('perc','groups'))
          except (RuntimeError, OSError, ValueError):
            pass

          h5.variables['perc'].long_name='percentile'
          h5.variables['perc'].units = '[ ]'
          h5.variables['perc'].axis = 'Y'
          h5.variables['groups'].long_name='time avg. group'
          h5.variables['groups'].units = 'min'
          h5.variables['groups'].axis = 'X'
          h5.variables['perc'][:] = perc
          h5.variables['groups'][:] = grnum
          h5.groups[resol].variables[varname].long_name = 'Rain-rate'
          h5.groups[resol].variables[varname].units = 'mm/h'
          

         

          for gr in groups:
              sys.stdout.flush()
              sys.stdout.write('\r Adding %s to %s' %(gr, os.path.basename(out)))
              sys.stdout.flush()
              name = 'P%s'%gr
              group = fn.groups[gr]
              if gr == '10min':
                  present = group.variables['ispresent'][:] > 0.
              else:
                  present = group.variables['isfile'][:] > thresh
              time = num2date(group.variables['time'][present].astype('i'), group.variables['time'].units)
              btime = burst.to_pydatetime()
              pr2 = [True if t in btime else False for t in time]
              data = np.ma.masked_outside(group.variables[varname][present,:][pr2],0.1,100000)
              DF[gr]= np.nanpercentile(data.filled(np.nan),perc)
              del data, present
              gc.collect()
          h5.groups[resol].variables[varname][:] = DF.values
          hdf5.put(resol, DF)


if __name__ == '__main__':
    filename = os.path.join(os.getenv('HOME'),'Data','Extremes','CPOL','CPOL_TIWI_1998-2017.nc')
    percs = list(range(0,100))+[99.9,99.99,99.999,99.9999,100]
    resol = 'all'
    for resol in ('all', 'break', 'burst'):
      print('Addeing %s'%resol)
      get_percentiles(filename, perc=percs, resol=resol)
