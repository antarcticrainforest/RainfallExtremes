import pandas as pd
from itertools import tee, repeat, chain, groupby
from datetime import  timedelta
import numpy as np

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
        group = ncf.groups[group]
        T = pd.DatetimeIndex(num2date(group.variables['time'][:],group.variables['time'].units))
        for t in range(len(T)):
            if group.variables['ispresent'][t] >= present:
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




