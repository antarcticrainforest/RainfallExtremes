import pandas as pd
from itertools import tee, repeat, chain, groupby
from datetime import  timedelta
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(os.path.basename(__file__))

def convScale(Series, method, freq, *args):
    """
    Convert a pandas timeseries of a frequency to another given (lower) frequency 
    by applying a given methond, like mean or sum

    Parameters:
        Series (pandas Series-object) : the input timeseries with freq. f1
        method (str-object)           : the method that is applied to create
                                        the new timeseries (e.g sum() or mean())
        freq (str-object)             : the desired frequency of the output
        *args                         : arguments that are passed to method 
                                        (if any)
    Returns : pandas Series-oject

    Example:
        >>> df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
        >>> dailySeries = df['AAPL.High']
        >>> dailySeries.index = df['Date']
        >>> freq = '1M'
        >>> monthlySeries = convScale(dailySeries, 'mean', freq)
    """

    # If not already make the index a DatetimeIndex
    Series.index = pd.DatetimeIndex(Series.index)

    return getattr(Series.groupby(pd.TimeGrouper(freq)),method)(*args)

def partition(Series):
    """
    Partition the timesteps of a given pandas timeseries by consecutive
    occurrance of its values

    Parameters:
        Series (pandas Series-object)  : input series

    Returns (dict-objct): Keys are the individual values of the input series,
                          Values is a list of pairs marking the start end end 
                          period of consecutive time periods

    Example:
        >>> df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
        >>> df.index = df.Date
        >>> df.direction.unique()
        ... array(['Increasing', 'Decreasing'], dtype=object)
        >>> table = partition(df.direction)
        >>> table.keys()
        ... dict_keys(['Increasing', 'Decreasing'])
        >>> table['Increasing'][6:8]
        ... [(Timestamp('2015-03-16 00:00:00'), Timestamp('2015-03-18 00:00:00')),
             (Timestamp('2015-03-23 00:00:00'), Timestamp('2015-03-23 00:00:00'))]
    """
    T = Series.index
    V = Series.values
    I = list(zip(T,V))
    sortkeyfn = key=lambda s:s[1]
    I.sort(key=sortkeyfn)
    result = {}
    for key,valuesiter in groupby(I, key=sortkeyfn):
        result[key]=np.array([v[0] for v in valuesiter])
    out = {}
    for key,indices in result.items():
        dates = []
        for start, end in datetime_range(pd.DatetimeIndex(indices).to_pydatetime()):
            dates.append((pd.Timestamp(start),pd.Timestamp(end)))
        out[key] = dates
    return out

def datetime_range(iterable):
    """
    Get all consecutive days for a list of dates

    Parameter:
        iterable : list or generator object of type pandas DatetimeIndex entries

    Returns generator with tuple marking start and end date of consecutive time period

    Example:
        >>> df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
        >>> periods = datetime_range(pd.DatetimeIndex(df.Dates))
        >>> list(periods)[5:8]
        ... [(Timestamp('2015-03-23 00:00:00'), Timestamp('2015-03-27 00:00:00')),
             (Timestamp('2015-03-30 00:00:00'), Timestamp('2015-04-02 00:00:00')),
             (Timestamp('2015-04-06 00:00:00'), Timestamp('2015-04-10 00:00:00'))]

    """
    iterable = sorted(set(iterable))
    keyfunc = lambda t: t[1] - timedelta(days=t[0])
    for key, group in groupby(enumerate(iterable), keyfunc):
        group = list(group)
        if len(group) == 1:
            yield group[0][1],group[0][1]
        else:
            yield group[0][1], group[-1][1]

def get_extremeTS(fname, group, thresh, present, varname='rain_rate'):
    '''
    Get a time series for extreme events of a given class of rainfall

    Arguments:
        faname  : Name of the netcdf file that stores the rainfall data
        thresh  : The group the data is stored
        perc    : Threshold in mm/h that defines an extreme event
        present : minimum percentage of 10min time-slices to be present to be considered
    Keyworkds:
        varname : variable name of the rain-rate
    '''
    from netCDF4 import Dataset as nc, num2date
    data = []
    index = []
    with nc(fname) as ncf:
        try:
            group = ncf.groups[group]
        except KeyError:
            group = ncf
        try:
            T = pd.DatetimeIndex(num2date(group.variables['time'][:],group.variables['time'].units))
        except KeyError:
            T = pd.DatetimeIndex(num2date(group.variables['t'][:],group.variables['t'].units))

        for t in range(len(T)):
            if 'ispresent' in group.variables.keys():
                vnpr = 'ispresent'
            else:
                vnpr = 'isfile'
            if group.variables[vnpr][t] >= present:
                index.append(t)
                if (group.variables[varname][t] >=  thresh).any():
                    data.append((T[t], 1))
                else:
                    data.append((T[t], 0))
    return pd.DataFrame(np.array(data),index=index,columns=['time','event'])



def get_shapes(colors, dates, opacity=0.2):
    """
    Create a list of plotly shapes to draw vertical line spans of given colors 
    for different values across given dates

    Parameters:
        colors (dict-object)  : The colors that each unique value should be assiged to
        date (dict-object)    : The dates (start, end) at which each object occurs
    Keywords:
        opacity               : alpha parameter of the facecolor

    Returns (list) shapes that can be passed to plotly

    Example:
        >>> df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
        >>> df.index = df.Date
        >>> df.direction.unique()
        ... array(['Increasing', 'Decreasing'], dtype=object)
        >>> table = partition(df.direction)
        >>> color = {'Increasing':'green','Decreasing':'red'}
        >>> shapes = get_shapes(color, table, opacity = 0.5)
        >>> import plotly.plotly as py
        >>> import plotly.graph_objs as go
        >>> import cufflinks as cf
        >>> cf.set_config_file(offline=False, world_readable=True, theme='ggplot')
        >>> data = [ go.Scatter(x=df['Date'],y=df['AAPL.Volume'].values) ]
        >>> layout = dict(shapes=shapes,yaxis=dict(title='Stock Price'))
        >>> py.iplot({'data': data, 'layout': layout}, filename='apple-stockprice')
    """
    shapes = []
    for regime, periods in dates.items():
        for start,end in periods:
            tmp =  {'type': 'rect',
                # x-reference is assigned to the x-values
                'xref': 'x',
                # y-reference is assigned to the plot paper [0,1]
                'yref': 'paper',
                'x0': start.strftime('%Y-%m-%d'),
                'y0': 0,
                'x1': end.strftime('%Y-%m-%d'),
                'y1': 1,
                'fillcolor': colors[regime],
                'opacity': opacity,
                'line': {
                    'width': 0,
                } }

            shapes.append(tmp)
    return shapes

def get_rainIndex(df, lon, lat, time, method='center'):
    '''
    Get all the indices for a raining centriod in a given grid

    Parameters:
        df (pd-dataframe) : dataframe containing the tracks
        lon (nd-array) : array containing the longitudes
        lat (nd-array) : array containing the latitudes

    Keywords:
        method: the index that is taken as the central point
                options (center: central point of the track
                         mean :  point with highest mean rain-rate
                         max  :  point with highest max rain-rate)

    Returns:
        index : array of inidces that represent the central storm of the track

    '''



    if method.lower() == 'center' or method.lower() == 'centre':
        idx = df.index[len(df) // 2]

    elif method.lower() == 'mean' or method.lower() == 'max':
        idx = df[method.lower()].idxmax()

    else:
        log.warning('Method not implemented falling back to center')
        idx = df.index[len(df) // 2]
    try:
        s_lon = np.fabs(lon.values - df.loc[idx]['lon']).argmin()
        s_lat = np.fabs(lat.values - df.loc[idx]['lat']).argmin()
    except :
        s_lon = np.fabs(lon - df.loc[idx]['lon']).argmin()
        s_lat = np.fabs(lat - df.loc[idx]['lat']).argmin()

    radius = df.loc[idx]['area'] // 2

    T = (pd.DatetimeIndex(time.values) - df.loc[idx].time).total_seconds()
    lons = (max(0, s_lon - radius), min(s_lon + radius, len(lon)))
    lats = (max(0, s_lat - radius), min(s_lat + radius, len(lat)))

    try:
        return np.fabs(T).argmin(), lats, lons
    except:
        return np.fabs(T).argmin(), lats, lons


if __name__ == '__main__':

    trackDir = '/home/unimelb.edu.au/mbergemann/Data/Extremes/UM/darwin/RA1T/20061109T1200Z/darwin/0p44km/Tracking'
    trackFile = os.path.join(trackDir,'tint_tracks_2006_11_09_12-2006_11_19_11.pkl')

    umFile = os.path.join('/home/unimelb.edu.au/mbergemann/Data/Extremes/UM/darwin/RA1T',
                          'um-0p44km-11091200-rain_20061109_1200-20061119_0600-2.5km.nc')

    import xarray as xr
    f = xr.open_dataset(umFile)
    lon = f.coords['lon']
    lat = f.coords['lat']
    t = f['lsrain']
    time = f['t']
    tracks = pd.read_pickle(trackFile)
    ii = 0
    for uid in np.unique(np.array(tracks.index.get_level_values('uid')).astype('i')):
        df = tracks.xs(str(uid), level='uid')
        tidx, lats, lons = get_rainIndex(df, lon, lat, time, 'center')
        print(t[tidx,0,lats[0]:lats[-1], lons[0]:lons[-1]].shape)
        ii += 1
        if ii > 3:
            break



