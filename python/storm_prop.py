from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from netCDF4 import date2num
import xarray as xr


def calc_thetae(T, p, q):
    Lv = 2.40e3
    kd = 0.2854
    cpd = 1005.7
    r = q/(1-q)
    p0 = 1024.
    return (T + Lv/cpd * r) * (p0/p)**kd


def calc_thetav(T, p, q):
        p0 = 1024.
        kd = 0.2854
        r = q/(1-q)
        theta = T * (p0/p)**kd
        return theta * (1 + 0.608*r)



def omega(files, loc, **kwargs):
    '''Get omega of a location'''
    tidx, loci, locj = loc #Index
    with xr.open_dataset(files['vert_wind']) as g:
        try:
            lons = g['lon'][locj[0]:locj[1]].values
            lats = g['lat'][loci[0]:loci[1]].values
        except KeyError:
            lons = g['longitude'][locj[0]:locj[1]].values
            lats = g['latitude'][loci[0]:loci[1]].values
        time = date2num(pd.DatetimeIndex(g['t'][tidx:tidx+1].values).to_pydatetime(),
                        'seconds since 1970-01-01 00:00:00')
        w = g['dz_dt'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    return w, lats, lons, np.array([time[0]])


def soilmoisture(files, loc, **kwargs):
    '''Get soilmoisture at a location'''
    try:
        tdelta = int(kwargs['tdelta'])
    except KeyError:
        tdelta = 1

    with xr.open_dataset(files['surf']) as f:
        tidx, loci, locj = loc #Index
        #Get the time windox from timedelta (tdelta in hours)
        time = pd.DatetimeIndex(f['t'][:].values)
        start = time[tidx] - timedelta(hours=tdelta)
        end = time[tidx] + timedelta(hours=tdelta)
        ttidx = np.where((time>= start)  & (time<= end))[0]
        try:
            lons = f['lon'][locj[0]:locj[1]].values
            lats = f['lat'][loci[0]:loci[1]].values
        except KeyError:
            lons = f['longitude'][locj[0]:locj[1]].values
            lats = f['latitude'][loci[0]:loci[1]].values

        t1, t2 = ttidx[0], ttidx[-1]+1

        time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                        'seconds since 1970-01-01 00:00:00')

        sm = f['sm'][t1:t2, 0, loci[0]:loci[1], locj[0]:locj[-1]].values
    return sm, lats, lons, np.array([time[0]])




def bflux(files, loc, **kwargs):
    '''Calculate Buoyancy flux'''
    try:
        tdelta = int(kwargs['tdelta'])
    except KeyError:
        tdelta = 1

    f, g = xr.open_dataset(files['vert_cent']), xr.open_dataset(files['vert_wind'])
    tidx, loci, locj = loc #Index

    #Get the time windox from timedelta (tdelta in hours)
    time = pd.DatetimeIndex(f['t'][:].values)
    start = time[tidx] - timedelta(hours=tdelta)
    end = time[tidx] + timedelta(hours=tdelta)
    ttidx = np.where((time>= start)  & (time<= end))[0]
    t1, t2 = ttidx[0], ttidx[-1]+1
    mt = (t2 - t1) // 2


    th = calc_thetav(f['temp'][t1:t2], f['p'].values.reshape(-1,1,1), f['q'][t1:t2])
    w = g['dz_dt'][t1:t2, :, loci[0]:loci[1], locj[0]:locj[-1]]
    b = th[:, :, loci[0]:loci[1], locj[0]:locj[1]] - th.mean(axis=(-2, -1))
    blx = (np.mean(b*w, axis=0) - np.mean(w, axis=0) * np.mean(b, axis=0)).values
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    f.close()
    g.close()
    return blx, lats, lons, np.array([time[0]])
    

