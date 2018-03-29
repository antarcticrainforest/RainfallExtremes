import numpy as np
from _radar import err
import os

def get(data, lon, lat, errorfile, percentile):
    '''
        Wrapper function that calls the C-extension to calculate radar errors
        Arguments:
            data (2D-Array)  = Array containing the radar data
            lon (1D-Array)   = Array containing the longitude vector
            lat (1D-Array)   = Array containing the latitude vector
            errrorfile       = File that contains the gauge statistics
            percentile       = the percentile that is considered
        Returns:
            2d array of radar-error

        Example:
            >>> import numpy as np
            >>> from netCDF4 import Dataset as nc
            >>> import radar_error
            >>> with nc('Radar_rain.nc') as rr:
            ...   data = rr.variables['rr'][2,:]
            ...   lon  = rr.variables['i'][:]
            ...   lat  = rr.variables['j'][:]
            >>> errfile = 'radar_error.txt'
            >>> perc = radar_error.get(data, lon, lat, errfile, 50.)
    '''
    shape = data.shape
    if not os.path.isfile(errorfile):
        raise RuntimeError('no such file %s'%errorfile)
    if len(data.shape) > 2:
        raise RuntimeError('rain-data array must be of rank 2')
    if len(lon.shape) != 1 or len(lon.shape) != 1:
        raise RuntimeError('lon/lat vectors must be of rank 1')
    try:
        data = np.ma.masked_invalid(data).filled(-99).ravel()
    except AttributeError:
        data = data.ravel()

    out =  np.ma.masked_equal(err(errorfile,
                                    data.astype(np.double),
                                    lat.astype(np.double),
                                    lon.astype(np.double),
                                    np.double(percentile)) ,-99)
    return np.ma.resize(out, shape)
