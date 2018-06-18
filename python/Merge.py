import os, sys, re, glob
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset as nc, num2date, date2num

class NCreate(nc):
    '''
     Wrapper class that creates a netcdf-file where the 10min, 3h, 6h and 24h
     radar-based rainfall is stored
    '''
    def __init__(self, *args, **kwargs):
        '''
        NC_create inherites from ncdf4.Dataset. The arguments and keywords are
        passed to the create instance. Please refer to the netCDF4.dataset doc
        for arguments and keywords
        '''
        nc.__init__(self, *args, **kwargs)
    def __enter__(self):
        ''' Return the netcdf-file object '''
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ''' Close the netcdf-file if done or something went wrong '''
        self.close()

    def create_var(self, meta, varname):
        '''
        Create all important variables like rain-rate, lon, lat and time
        Arguments:
            meta : dict with information about additional info
                    (size of the time vector per day, time units, missing_values)
            cpol : mask that represents the cpol disk
            twp  : mask that represents the twp area

        '''
        chunks_shape = []
        var = []
        for i in meta['dims'].keys():
            chunks_shape.append(len(meta['dims'][i]['data']))
            self.createDimension(i,len(meta['dims'][i]['data']))
            self.createVariable(i, 'f', (i, ))
            self.variables[i].long_name=meta['dims'][i]['name']
            self.variables[i].units=meta['dims'][i]['units']
            self.variables[i][:]=meta['dims'][i]['data']
            var.append(i)

        chunks_shape.append(meta['size'])
        var.append('time')
        self.createDimension('time', None)
        self.createVariable('time', 'i', ('time', ))

        self.variables['time'].units = meta['units']
        self.variables['time'].axis = 'T'
        self.variables['time'].long_name = 'time'
        self.variables['time'].calendar = 'Standard'
        self.variables['time'].short_name = 't'
        self.variables['time'].standard_name = 'time'

        self.createVariable(varname, 'f', tuple(var),
                            fill_value=meta['missing'], zlib=True,
                            complevel=3, least_significant_digit=4,
                            chunksizes=tuple(chunks_shape))
        self.variables[i].units = meta['var']['units']
        self.variables[i].long_name = meta['var']['name']
        self.variables[i].missing_value = meta['missing']



def get_files(directory, tstep, ending):
    '''
    Get filename of radar files in a given data period
    Arguments:
        directory: The parent directory where data is stored
        tstep : Frist date of the period
        ending : Last date of the period
    Returns:
        List of filenames

    '''
    files = []
    for year in range(tstep.year, ending.year+1):
        if len(glob.glob(os.path.join(directory,'*.nc'))):
            #All files are in one folder
            for fn in glob.glob(os.path.join(directory,'*.nc')):
                files.append(os.path.join(directory, os.path.basename(fn)))
            files.sort()
            return files
        elif len(glob.glob(os.path.join(directory,'%04i'%year,'*.nc'))):
            #Files are sorted per year
            for fn in glob.glob(os.path.join(directory,'%04i'%year,'*.nc')):
                files.append(os.path.join(directory,'%04i'%year,
                             os.path.basename(fn)))
        else:
            #Files are organized in daily folders
            for subdir in glob.glob(os.path.join(directory,'%04i'%year,'????????')):
                files+= [fn for fn in glob.glob(os.path.join(subdir,'*.nc'))]

    files.sort()
    return files



def main(datafolder, first, last, out, dims, metafile=None,
         varname='radar_estimated_rain_rate', outvar='rain_rate'):
    '''
    This function gets radar rainfall data, stored in daily files and stores it
    in a netcdf-file where all the data is stored. It also calculates 1, 3, 6 and
    24 hourly averages of rainfall from the 10 minute input fields

    Arguments:
        datafoler (str)  : the parent directory where the data is stored
        first (datetime) : first month of the data (YYYYMM)
        last  (datetime) : last month of the data (YYYYMM)
        out (str)        : The filname of the output data
        dims (tuple)     : tuple of dimension names (str)
    Returns:
        None
    '''
    files = get_files(datafolder, first, last)
    meta = dict(dims={})
    #Get meta data for this file
    if not ( metafile is None):
      mfile = metafile
    else:
      mfile = files[0]
    name = {'longitude':'longitude', 'latitude':'latitude', 'time':'time'}
    with nc(mfile) as fnc:
        for d in dims:
            meta['dims'][d] = {}
            meta['dims'][d]['data'] = fnc.variables[d][:]
            try:
              meta['dims'][d]['name'] = fnc.variables[d].long_name
            except AttributeError:
              meta['dims'][d]['name'] = name[d]
            meta['dims'][d]['units']= fnc.variables[d].units
        meta['var']= dict(name=fnc.variables[varname].long_name,
                          units=fnc.variables[varname].units)
        meta['units'] = 'seconds since 1970-01-01 00:00:00'
        meta['missing'] = -9999.0
        try:
          meta['size'] = fnc.dimensions['time'].size
        except KeyError:
          meta['size'] = 144
    def get_tr(num):
        return tuple(list(range(1,num))+[0])
    with NCreate(out, 'w', dist_format='NETCDF4', disk_format='HDF5') as fnc:
        fnc.create_var(meta, outvar)
        for tt, fname in enumerate(files):
            with nc(fname) as source:
                sys.stdout.flush()
                sys.stdout.write('\rAdding %s'%(os.path.basename(fname)))
                sys.stdout.flush()
                # Read data
                rain_rate =  np.ma.masked_invalid(source.variables[varname][:])
                #Check if data has to be 'appended' or adde (first time step)
                if tt == 0:
                    size = 0
                else:
                    size = fnc.dimensions['time'].size
                timevar=num2date(source.variables['time'][:],source.variables['time'].units)
                try:
                  fnc.variables[outvar][...,size:] = rain_rate.transpose(get_tr(len(dims)+1))
                  fnc.variables['time'][size:] = date2num(timevar, meta['units'])
                except IndexError:
                  pass
        sys.stdout.write('\n')

if __name__ == '__main__':

    starting = '19981206'
    ending = '20170502'
    #
    maskfile = None
    dim = ('lon', 'lat')
    
    #datadir = os.path.join(os.getenv('HOME'), 'Data', 'CPOL', 'netcdf', 'PPI')
    #datadir = '/g/data2/rr5/vhl548/CPOL_level_1b/PPI'
    datadir = '/g/data2/rr5/vhl548/NEW_CPOL_level_2/RADAR_ESTIMATED_RAIN_RATE'
    metafile = os.path.join(os.getenv('HOME'), 'Data', 'Extremes', 'CPOL', 'CPOL_masks.nc')
    outfile = os.path.join(os.getenv('HOME'),'Data', 'Extremes','CPOL','CPOL_1998-2017.nc')
    main(datadir, datetime.strptime(starting, '%Y%m%d'),
         datetime.strptime(ending, '%Y%m%d'), outfile, dim, metafile=metafile)
