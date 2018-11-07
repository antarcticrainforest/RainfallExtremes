from datetime import datetime, timedelta
import sys

import pandas as pd
import numpy as np
from netCDF4 import date2num
import xarray as xr


def calc_thetap(T, p, q, rr):
    p0 = 1000.
    kd = 0.2854
    r = q/(1-q)
    qr = rr/5e3
    theta = T * (p0/p)**kd
    g = 9.81

    return theta * (1 + 0.608*r - qr)

def calc_thetae(T, p, q):
    Lv = 2.40e3
    kd = 0.2854
    cpd = 1005.7
    r = q/(1-q)
    p0 = 1000.
    return (T + Lv/cpd * r) * (p0/p)**kd


def calc_thetav(T, p, q):
        p0 = 1000.
        kd = 0.2854
        r = q/(1-q)
        theta = T * (p0/p)**kd
        return theta * (1 + 0.608*r)

def calc_rho(q, T, p):
    Rv = 461.495
    Rd = 287.058

    es = 6.112 * np.exp( (17.62 * (T - 273.15)) / (243.12 - (T - 273.15)))
    s = 0.622 * es / p
    #s[s==0] = 1e-6
    rh = q / s
    rh[rh<0] = 0
    rh[rh>1] = 1
    pv = rh * es
    pd = p - pv
    rho = pd/(Rd*T) + pv/(Rv*T)
    return rho

def q(files, loc, **kwargs):
    '''Get humidity of a location'''
    tidx, loci, locj = loc #Index
    with xr.open_dataset(files['vert_cent']) as g:
        try:
            lons = g['lon'][locj[0]:locj[1]].values
            lats = g['lat'][loci[0]:loci[1]].values
        except KeyError:
            lons = g['longitude'][locj[0]:locj[1]].values
            lats = g['latitude'][loci[0]:loci[1]].values
        time = date2num(pd.DatetimeIndex(g['t'][tidx:tidx+1].values).to_pydatetime(),
                        'seconds since 1970-01-01 00:00:00')
        q = g['q'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values -\
                g['q'][tidx].mean(axis=(-2,-1)).values.reshape(-1,1,1)
    return q, lats, lons, np.array([time[0]])




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
        mw = g['dz_dt'][tidx].mean(axis=(-2,-1)).values

    return w - mw.reshape(-1,1,1), lats, lons, np.array([time[0]])


def soilmoisture(files, loc, **kwargs):
    '''Get soilmoisture at a location'''
    try:
        tdelta = int(kwargs['tdelta'])
    except KeyError:
        tdelta = 4

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


def mflux(files, loc, **kwargs):
    '''Calculate moisture flux'''
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


    w = g['dz_dt'][t1:t2, :, loci[0]:loci[1], locj[0]:locj[-1]]
    q = f['q'][t1:t2, :, loci[0]:loci[1], locj[0]:locj[1]]
    mflux = (np.mean(q*w, axis=0) - np.mean(w, axis=0) * np.mean(q, axis=0)).values
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
    return mflux, lats, lons, np.array([time[0]])

def mfluxdiv(files, loc, **kwargs):
    '''Calculate moisture flux divergence'''
    try:
        tdelta = int(kwargs['tdelta'])
    except KeyError:
        tdelta = 1

    f, g = xr.open_dataset(files['vert_cent']), xr.open_dataset(files['vert_wind'])
    tidx, loci, locj = loc #Index
    P1=(float(g.variables['lon'][0]), float(g.variables['lat'][0]))
    P2=(float(g.variables['lon'][0]), float(g.variables['lat'][1]))
    dx = cdist(P1,P2).round(3)*1000
    #Get the time windox from timedelta (tdelta in hours)
    time = pd.DatetimeIndex(f['t'][:].values)
    start = time[tidx] - timedelta(hours=tdelta)
    end = time[tidx] + timedelta(hours=tdelta)
    ttidx = np.where((time>= start)  & (time<= end))[0]
    t1, t2 = ttidx[0], ttidx[-1]+1
    mt = (t2 - t1) // 2


    q = f['q'][mt, :, loci[0]:loci[1], locj[0]:locj[1]]# - f['q'][mt].mean(axis=(-2,-1))
    v = g['v'][mt, :, loci[0]:loci[1], locj[0]:locj[1]]# - g['v'][mt].mean(axis=(-2,-1))
    u = g['u'][mt, :, loci[0]:loci[1], locj[0]:locj[1]]# - g['u'][mt].mean(axis=(-2,-1))

    qvp = (np.mean(u*q, axis=(-2,-1)) - np.mean(q, axis=(-2,-1)) * np.mean(q, axis=(-2,-1))).values
    qup = (np.mean(v*q, axis=(-2,-1)) - np.mean(q, axis=(-2,-1)) * np.mean(q, axis=(-2,-1))).values
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

    dqvp = np.gradient(q*v*1000, dx, axis=(-1,-2))[0]
    dqup = np.gradient(q*u*1000, dx, axis=(-1,-2))[1]

    return -(dqvp + dqup), lats, lons, np.array([time[0]])

def cdist(p1, p2):
## approximate radius of earth in km
    R = 6373.0
    lat1 = np.radians(p1[1])
    lon1 = np.radians(p1[0])
    lat2 = np.radians(p2[1])
    lon2 = np.radians(p2[0])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return  R * c

def ctt(files, loc, **kwargs):
    from model2lev import interp
    '''Get cloud top temperature of a location'''
    tidx, loci, locj = loc #Index
    with xr.open_dataset(files['surf']) as g:
        try:
            lons = g['lon'][locj[0]:locj[1]].values
            lats = g['lat'][loci[0]:loci[1]].values
        except KeyError:
            lons = g['longitude'][locj[0]:locj[1]].values
            lats = g['latitude'][loci[0]:loci[1]].values
        time = date2num(pd.DatetimeIndex(g['t'][tidx:tidx+1].values).to_pydatetime(),
                        'seconds since 1970-01-01 00:00:00')
        ctt = g['ctt'][tidx, loci[0]:loci[1], locj[0]:locj[-1]].values
    return ctt, lats, lons, np.array([time[0]])
    dqup = np.gradient(qup[1], axis=0)

    return -(dqvp + dqup)

def cloud_p(files, loc, **kwargs):
    from model2lev import interp
    f, g = xr.open_dataset(files['qcl_th']), xr.open_dataset(files['qcf_th'])
    p = xr.open_dataset(files['res_th'])
    Pf = xr.open_dataset(files['vert_cent'])
    P = Pf.variables['p'].values
    Pf.close()

    tidx, loci, locj = loc #Index
    qcl = f['QCL'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    qcf = g['QCF'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    pres = p['p'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    f.close(), g.close(), p.close()
    return interp(qcl+qcf, pres/100, P), lats, lons, np.array([time[0]])

def cloud_z(files, loc, **kwargs):
    from model2lev import interp
    f, g = xr.open_dataset(files['qcl_th']), xr.open_dataset(files['qcf_th'])
    z = xr.open_dataset(files['geop_th'])
    Z = np.linspace(5, 9000, 22)

    tidx, loci, locj = loc #Index
    qcl = f['QCL'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    qcf = g['QCF'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    ht = z['ht'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    f.close(), g.close(), z.close()
    return interp(qcl+qcf, ht, Z), lats, lons, np.array([time[0]])




def momflux(files, loc, **kwargs):
    '''Calculate momentumflux'''
    try:
        tdelta = int(kwargs['tdelta'])
    except KeyError:
        tdelta = 1

    f, g = xr.open_dataset(files['vert_cent']), xr.open_dataset(files['vert_wind'])
    tidx, loci, locj = loc #Index

    # Get the time windox from timedelta (tdelta in hours)
    time = pd.DatetimeIndex(f['t'][:].values)
    start = time[tidx] - timedelta(hours=tdelta)
    end = time[tidx] + timedelta(hours=tdelta)
    ttidx = np.where((time>= start)  & (time<= end))[0]
    t1, t2 = ttidx[0], ttidx[-1]+1
    mt = (t2 - t1) // 2


    w = g['dz_dt'][t1:t2, :, loci[0]:loci[1], locj[0]:locj[-1]]
    v = g['v'][t1:t2, :, loci[0]:loci[1], locj[0]:locj[1]]
    u = g['u'][t1:t2, :, loci[0]:loci[1], locj[0]:locj[1]]
    uwp = (np.mean(u*w, axis=0) - np.mean(w, axis=0) * np.mean(u, axis=0)).values
    vwp = (np.mean(v*w, axis=0) - np.mean(w, axis=0) * np.mean(v, axis=0)).values
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    rho = calc_rho(f['q'][tidx,:, loci[0]:loci[1], locj[0]:locj[1]].values,
                   f['temp'][tidx,:, loci[0]:loci[1], locj[0]:locj[1]].values,
                   f['p'].values.reshape(1,-1,1,1))

    f.close()
    g.close()
    #return 9.81*rho*np.sqrt(uwp**2+vwp**2), lats, lons, np.array([time[0]])
    return uwp, lats, lons, np.array([time[0]])


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
    

