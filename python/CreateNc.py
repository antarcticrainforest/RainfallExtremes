import os,sys
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
        size = self.dimensions['time-%ih'%hours].size
        self.variables['rain_rate-%ih'%hours][size:, :] = data
        self.variables['ispresent-%ih'%hours][size:] = isfile / times.shape[1]
        self.variables['time-%ih'%hours][size:] = times[:,0]

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
                add_var, time_avg = '-%ih'%avgtype, '%2i hly' %avgtype
                present_unit = '%'
            else:
                add_var, time_avg = '', '10 Min'
                present_unit = 'bool'

            self.createDimension('time%s'%add_var, None)
            self.createVariable('time%s'%add_var, 'i', ('time%s'%add_var, ))
            self.createVariable('rain_rate%s'%add_var, 'f',
                                ('time%s'%add_var, 'lat', 'lon'), fill_value=meta['missing'],
                                zlib=True, complevel=9, least_significant_digit=4,
                                chunksizes=chunks_shape)

            self.variables['time%s'%add_var].units = meta['units']
            self.variables['time%s'%add_var].axis = 'T'
            self.variables['time%s'%add_var].long_name = 'time'
            self.variables['time%s'%add_var].calendar = 'Standard'
            self.variables['time%s'%add_var].short_name = 't'
            self.variables['time%s'%add_var].standard_name = 'time'

            self.variables['rain_rate%s'%add_var].units = 'mm/h'
            self.variables['rain_rate%s'%add_var].standard_name = '%s rain_rate' %time_avg
            self.variables['rain_rate%s'%add_var].short_name = 'rr'
            self.variables['rain_rate%s'%add_var].long_name = '%s Rain Rate '%time_avg
            self.variables['rain_rate%s'%add_var].missing_value = meta['missing']

            self.createVariable('ispresent%s'%add_var, 'f', ('time%s'%add_var, ))
            self.variables['ispresent%s'%add_var].long_name = 'Present time steps (%s)'%time_avg
            self.variables['ispresent%s'%add_var].short_name = '%s present'%time_avg
            self.variables['ispresent%s'%add_var].standard_name = 'present time step %s'%time_avg
            self.variables['ispresent%s'%add_var].units = present_unit

        self.createVariable('contours', 'i', ('time', 'lat', 'lon'),
                            fill_value=meta['missing'], zlib=True, complevel=9,
                            least_significant_digit=2, chunksizes=chunks_shape)

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

        self.variables['contours'].units = ' '
        self.variables['contours'].standard_name = 'contours'
        self.variables['contours'].short_name = 'cnt'
        self.variables['contours'].long_name = 'Contours of rainfall areas'
        self.variables['contours'].missing_value = meta['missing']


        for i in ('1h', '3h'):
            self.createVariable('contours-%s'%i, 'i', ('time-%s'%i, 'lat', 'lon'),
                                fill_value=meta['missing'], zlib=True, complevel=9,
                                least_significant_digit=2, chunksizes=chunks_shape)
            self.variables['contours-%s'%i].units = ' '
            self.variables['contours-%s'%i].standard_name = 'contours %sly'%i
            self.variables['contours-%s'%i].short_name = 'cnt %sly'%i
            self.variables['contours-%s'%i].long_name = '%sly Contours of rainfall areas'
            self.variables['contours-%s'%i].missing_value = meta['missing']


def main(datafolder, first, last, out, timeavg=(1, 3, 6, 24)):
    '''
    This function gets radar rainfall data, stored in daily files and stores it
    in a netcdf-file where all the data is stored. It also calculates 1, 3, 6 and
    24 hourly averages of rainfall from the 10 minute input fields

    Arguments:
        datafoler (str)  : the parent directory where the data is stored
        first (datetime) : first month of the data (YYYYMM)
        last  (datetime) : last month of the data (YYYYMM)
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


    with NCreate(out, 'w', dist_format='NETCDF4', disk_format='HDF5') as fnc:
        fnc.create_rain(lat, lon, list(timeavg), meta)
        for tt, fname in enumerate(files):
            with nc(fname) as source:
                sys.stdout.flush()
                sys.stdout.write('Adding %s ... '%(os.path.basename(fname)))
                sys.stdout.flush()

                try:
                     rain_rate =  np.ma.masked_invalid(source.variables['radar_estimated_rain_rate'][:].filled(0))
                except AttributeError:
                     rain_rate = np.ma.masked_invalid(source.variables['radar_estimated_rain_rate'][:])
                if tt == 0:
                    size = 0
                else:
                    size = fnc.dimensions['time'].size
                fnc.variables['time'][size:] = source.variables['time'][:]
                fnc.variables['rain_rate'][size:, :] = rain_rate
                fnc.variables['ispresent'][size:] = source.variables['isfile'][:]
                for hour in timeavg:
                    data, hsize = fnc.create_avg(rain_rate,source.variables['isfile'][:],
                                           source.variables['time'][:], hour)
                    if hour in (1,3):
                        contours = np.zeros_like(data)
                        for i in range(len(data)):
                            contours[i] = get_countours(np.ma.masked_less(data[i], 2))
                        fnc.variables['contours-%ih'%hour][hsize:, :] = np.ma.masked_equal(contours,0)

                contours = np.zeros_like(rain_rate)
                for i in range(len(rain_rate)):
                    contours[i] = get_countours(np.ma.masked_less(rain_rate[i], 2))

                fnc.variables['contours'][size:, :] = np.ma.masked_equal(contours,0)
            sys.stdout.write('ok\n')

if __name__ == '__main__':

    starting = '19981206'
    ending = '20170502'
    #
    datadir = os.path.join(os.getenv('HOME'), 'Data', 'Darwin', 'netcdf')
    main(datadir, datetime.strptime(starting, '%Y%m%d'),
         datetime.strptime(ending, '%Y%m%d'), os.path.join(datadir, 'CPOL.nc'))
