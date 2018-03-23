import os, sys, re
import numpy as np
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter as bl
from netCDF4 import Dataset as nc

def get_countours(array, kernel_size=3, sigma=0.15):
    '''
        Apply Canny edge detection on rainfall data
        Arguments:
            array (2d float) : 2D Array containing the rainfall data
        Keywords:
            kernle_size  = the size of the stucturing element
            sigma = sigma value for the Gaussian filter
        Returns:
            2D array with labeled contours of rainfall
    '''
    import cv2
    #Try to make masked values 0
    try:
        array = np.ma.masked_invalid(array).filled(0)
    except AttributeError:
        pass
    #Create a unit array
    array[array != 0] = 255
    bw = array.astype(np.uint8)
    #Now detect the edges of the black-and-white image
    #But first close small holes with erosion and dilation
    #For this purpose we need a kernel (this case a cross)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,\
                                        (kernel_size-1, kernel_size-1))
    #erosion and dilation
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, (kernel_size + 5,
                             kernel_size + 5))
    for c in range(3):
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                                 (kernel_size + 5, kernel_size + 5))

    #blur the closed-hole image
    closed = bl(closed, sigma)
    #Api-change in from openCV 2.x to 3.x check for major version
    cv2vers = int(cv2.__version__[0])
    #Now get the contours with canny-edge detection
    out = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Now just get the number of coastal features
    if cv2vers < 3:
        contours, hierarchy = out
    else:
        _, contours, hierarchy = out
    # Output array
    mask = np.zeros(closed.shape, dtype='uint8')

    hiern = 1
    for nn, cntour in enumerate(contours):
        if hierarchy[0][nn][-1] == -1:
            cv2.drawContours(mask, [cntour], -1, hiern, -1)
            hiern += 1
    return mask.astype('int32')

def get_files(directory, tstep, ending,
              fstring='CPOL_RADAR_ESTIMATED_RAIN_RATE_%Y%m%d_level2.nc'):
    '''
    Get filename of radar files in a given data period
    Arguments:
        directory: The parent directory where data is stored
        tstep : Frist date of the period
        ending : Last date of the period
    Keywords:
        fstring: Filename of each daily file, should also contain the data
        format for the date
    Returns:
        List of filenames

    '''
    files = []
    while tstep <= ending:
        fname = os.path.join(directory, '%04i'%tstep.year, tstep.strftime(fstring))
        if os.path.isfile(fname):
            files.append(fname)
        tstep += timedelta(hours=24)
    files.sort()
    return files

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

    def create_avg(self, data, isfile, times, hours):
        '''
        Create a avg over an amount of given hours
        Arguments:
            data (3D-array) : datafield containg the original data
            isfile (1D-array) : infromation (1/0) about available timesteps
            times (1D-array) : timestep information
            hours (int) : the averaging period
        Returns:
            3d-array avg of data
        '''
        #Check if for consitency
        if 24 % hours != 0 or data.shape[0] % (24/hours) != 0 :
            raise RuntimeError('Cannot split dataset into even number of chunks')

        data = np.ma.masked_array(np.split(data, int(24 / hours))).mean(axis=1)
        isfile = np.array(np.split(isfile, int(24 / hours))).sum(axis=1) * 100.
        times = np.array(np.split(times, int(24 / hours)))
        size = self['%ih'%hours].dimensions['time'].size
        self['%ih'%hours].variables['rain_rate'][size:, :] = data
        self['%ih'%hours].variables['rain_rate-flip'][...,size:] = data.transpose(1,2,0)
        self['%ih'%hours].variables['ispresent'][size:] = isfile / times.shape[1]
        self['%ih'%hours].variables['time'][size:] = times[:,0]

        return data, size

    def create_rain(self, lat, lon, times, meta):
        '''
        Create all important variables like rain-rate, lon, lat and time
        Arguments:
            lat : 1D - Array : Latitude vector
            lon : 1D - Array : Longitude vector
            times : tuple containing int's marking average periods (in hours)
            meta : dict with information about additional info
                    (size of the time vector per day, time units, missing_values)

        '''
        chunks_shape = (meta['size'], len(lon), len(lat))
        self.createDimension('lon', len(lon))
        self.createDimension('lat', len(lat))
        self.createVariable('lon', 'f', ('lon', ))
        self.createVariable('lat', 'f', ('lat', ))
        for avgtype in [None]+times:
            if isinstance(avgtype, int):
                group_name = '%ih'%avgtype
                present_unit = '%'
            else:
                group_name = '10min'
                present_unit = 'bool'
            group = self.createGroup(group_name)
            group.createDimension('time', None)
            group.createVariable('time', 'i', ('time', ))

            group.variables['time'].units = meta['units']
            group.variables['time'].axis = 'T'
            group.variables['time'].long_name = 'time'
            group.variables['time'].calendar = 'Standard'
            group.variables['time'].short_name = 't'
            group.variables['time'].standard_name = 'time'

            for i, d in (('rain_rate', ('time', 'lat', 'lon')),
                          ('rain_rate-flip', ('lat', 'lon', 'time'))):
                if i == 'rain_rate':
                    ch = chunks_shape
                else:
                    ch = (chunks_shape[1],chunks_shape[2],chunks_shape[0])
                group.createVariable(i, 'f', d,
                                     fill_value=meta['missing'], zlib=False,
                                     complevel=9, least_significant_digit=4,
                                     chunksizes=ch)
                group.variables[i].units = 'mm/h'
                group.variables[i].standard_name = 'rain_rate'
                group.variables[i].short_name = 'rr'
                group.variables[i].long_name = 'Rain Rate '
                group.variables[i].missing_value = meta['missing']

            group.createVariable('ispresent', 'f', ('time', ))
            group.variables['ispresent'].long_name = 'Present time steps'
            group.variables['ispresent'].short_name = 'present'
            group.variables['ispresent'].standard_name = 'present timesteps'
            group.variables['ispresent'].units = present_unit
            if avgtype in (None, 1, 3):
                group.createVariable('contours', 'i', ('time', 'lat', 'lon'),
                                     fill_value=meta['missing'], zlib=False, complevel=9,
                                     least_significant_digit=2, chunksizes=chunks_shape)
                group.variables['contours'].units = ' '
                group.variables['contours'].standard_name = 'contours'
                group.variables['contours'].short_name = 'cnt'
                group.variables['contours'].long_name = 'Contours of rainfall areas'
                group.variables['contours'].missing_value = meta['missing']



        self.variables['lon'][:] = lon
        self.variables['lon'].units = 'degrees_east'
        self.variables['lon'].axis = 'X'
        self.variables['lon'].long_name = 'Longitude'
        self.variables['lon'].standard_name = 'longitude'
        self.variables['lon'].short_name = 'lon'
        self.variables['lat'].units = 'degrees_north'
        self.variables['lat'].axis = 'Y'
        self.variables['lat'].long_name = 'Latitude'
        self.variables['lat'].standard_name = 'latitude'
        self.variables['lat'].short_name = 'lat'
        self.variables['lat'][:] = lat
    @staticmethod
    def getSplit(length, maxn = 2**8):
        from functools import reduce
        ''' Ruturn the optimal chunk value for an array '''
        n = np.array(reduce(list.__add__,
                            ([i, length//i] for i in range(1, int(pow(length, 0.5) + 1)) if length % i == 0)))
        n.sort()
        return n[np.fabs(n-maxn).argmin()]

    def transpose_var(self, varname, group=None, dims=('lat', 'lon', 'time')):
        '''
        Method to copy an existing netCDF variable in reverse order

        Arguments:
            varname : name of the variable

        Keywords:
            group : optional group where the variable is stored (default None)
            dims : the desired shape of the output variable

        Returns:

        '''
        if isinstance(group, type(None)):
            group = self
        else:
            group = self[group]

        ncvar = group.variables[varname]

        sys.stdout.flush()
        sys.stdout.write('\rFlipping dimensions to %s of %s ... done  '%(dims, varname))
        sys.stdout.flush()

        dim1 = list(dims)
        dim2 = list(ncvar.dimensions)
        dim1.sort(), dim2.sort()
        if dim1 != dim2:
            raise RuntimeError('Dimensions names of variables must be identical')
        #Get the dimension of the variable
        reorder = tuple(ncvar.dimensions.index(i) for i in dims)
        chunks = [ncvar.chunking()[i] for i in reorder]

        #Now create the new variable
        try:
            group.createVariable(varname+'-flip',ncvar.dtype, dims, zlib=False,
                                 complevel=9, chunksizes=chunks, fill_value=ncvar.missing_value,
                                 least_significant_digit=ncvar.least_significant_digit)
        except (RuntimeError, IndexError, AttributeError):
            pass
        for attr in ncvar.ncattrs():
            if 'fillvalue' not in attr.lower():
                setattr(group.variables[varname+'-flip'], attr, getattr(ncvar,attr))

        split = getSplit(ncvar.shape[0], )
        index = list(range(0,ncvar.shape[0]+split,split))
        for i in range(len(index)-1):
            sys.stdout.write('\rFlipping dimensions to %s of %s ... %3i%%'%(dims, varname, float(i)*100/len(index) ))
            sys.stdout.flush()
            group.variables[varname+'-flip'][...,index[i]:index[i+1]] = ncvar[index[i]:index[i+1]].transpose(reorder)
        sys.stdout.write('\rFlipping dimensions to %s of %s ... done  '%(dims,varname))
        sys.stdout.flush()
        sys.stdout.write('\n')
def get_prev(f1, varname, mask, hour):
    '''
        Get data from the previous time-step if present, if not just return mask
        values

        Arguments:
            f1 : The filename of the current file
            varname : variable name of the rainfall data
            mask : mask that is applied to the data
            hour : hourly avg
    '''
    dirname = os.path.dirname(f1)
    fname = os.path.basename(f1)
    date = re.findall(r'\d{8}', fname)[0]
    previous = datetime.strptime(date,'%Y%m%d') - timedelta(seconds=(hour*60**2)/2)
    if previous.month == 12 and previous.day == 31:
        dirname = dirname.replace(str(previous.year+1), str(previous.year))
    previous_file = os.path.join(dirname, fname.replace(date,previous.strftime('%Y%m%d')))

    length = int(hour*60/2 / 10)
    if os.path.isfile(previous_file):
        with nc(previous_file) as gnc:
            rr = gnc.variables[varname][-length:]
            rr = np.ma.masked_invalid(mask*np.ma.masked_less(rr, 0.1).filled(0))
            ifile = gnc.variables['isfile'][-length:]
    else:
        rr = np.ma.masked_less(np.zeros([length, mask.shape[0], mask.shape[0]]), 1)
        ifile = np.zeros([length])
    return (ifile, rr)
def concate(r, isf, prev_r, prev_isf):
    '''
    Concat data array
    '''

    return np.ma.masked_invalid(np.ma.concatenate((prev_r, r))[:-len(prev_r)]),\
            np.ma.concatenate((prev_isf, isf))[:-len(prev_isf)]
def main(datafolder, first, last, maskfile, out, timeavg=(1, 3, 6, 24)):
    '''
    This function gets radar rainfall data, stored in daily files and stores it
    in a netcdf-file where all the data is stored. It also calculates 1, 3, 6 and
    24 hourly averages of rainfall from the 10 minute input fields

    Arguments:
        datafoler (str)  : the parent directory where the data is stored
        first (datetime) : first month of the data (YYYYMM)
        last  (datetime) : last month of the data (YYYYMM)
        maskfile         : Additional mask applied to the data
        out (st)         : The filname of the output data
    Keywords:
        timeavg (list)   : list containing averaging periods
    Returns:
        None
    '''
    from glob import glob
    files = get_files(datafolder, first, last)
    meta = {}
    with nc(files[0]) as fnc:
        lon = fnc.variables['longitude'][0, :]
        lat = fnc.variables['latitude'][:, 0]
        meta['units'] = fnc.variables['time'].units
        meta['missing'] = -9999.0
        meta['size'] = fnc.dimensions['time'].size
    if not isinstance(type(maskfile),type(None)):
        with nc(maskfile) as fnc:
            mask = np.ma.masked_invalid(fnc.variables['mask_ring'][:])
    else:
        mask = np.ones([len(lat), len(lon)])
    varname = 'radar_estimated_rain_rate'
    with NCreate(out, 'w', dist_format='NETCDF4', disk_format='HDF5') as fnc:
        fnc.create_rain(lat, lon, list(timeavg), meta)
        for tt, fname in enumerate(files):
            with nc(fname) as source:
                sys.stdout.write('\rAdding %s ... '%(os.path.basename(fname)))
                sys.stdout.flush()

                rain_rate =  np.ma.masked_invalid(mask * np.ma.masked_less(
                                                  source.variables[varname][:],0.1).filled(0))
                if tt == 0:
                    size = 0
                else:
                    size = fnc['10min'].dimensions['time'].size
                fnc['10min'].variables['time'][size:] = source.variables['time'][:]
                fnc['10min'].variables['rain_rate'][size:, :] = rain_rate
                fnc['10min'].variables['rain_rate-flip'][...,size:] = rain_rate.transpose(1,2,0)
                fnc['10min'].variables['ispresent'][size:] = source.variables['isfile'][:]
                for hour in timeavg:
                    prev_isfile, prev_rr = get_prev(fname, varname, mask, hour)
                    rr, isfile = concate(rain_rate, source.variables['isfile'][:], prev_rr, prev_isfile)
                    data, hsize = fnc.create_avg(rr, isfile, source.variables['time'][:],
                                                 hour)
                    if hour in (1,3):
                        contours = np.zeros_like(data)
                        for i in range(len(data)):
                            contours[i] = get_countours(np.ma.masked_less(data[i], 2))
                        fnc['%ih'%hour].variables['contours'][hsize:, :]\
                                = np.ma.masked_equal(contours,0)

                contours = np.zeros_like(rain_rate)
                for i in range(len(rain_rate)):
                    contours[i] = get_countours(np.ma.masked_less(rain_rate[i], 2))

                fnc['10min'].variables['contours'][size:, :] = np.ma.masked_equal(contours,0)
            sys.stdout.write('\rAdding %s ... ok'%(os.path.basename(fname)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        #Create a copy of all rain-rate variables (lat,lon,time) order
        #for i in ('10min','1h','3h','6h','24'):
        #    fnc.transpose_var('rain_rate', group=i)

if __name__ == '__main__':

    starting = '19981206'
    ending = '20170502'
    #
    maskfile = os.path.join(os.getenv('HOME'), 'Data', 'Darwin', 'netcdf','cpol_mask_ring.nc')
    datadir = os.path.join(os.getenv('HOME'), 'Data', 'Darwin', 'netcdf')
    main(datadir, datetime.strptime(starting, '%Y%m%d'),
         datetime.strptime(ending, '%Y%m%d'), maskfile, os.path.join(datadir, 'CPOL.nc'))
