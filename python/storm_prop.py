from datetime import datetime, timedelta
import sys

import pandas as pd
import numpy as np
from netCDF4 import date2num
import xarray as xr

np.warnings.filterwarnings('ignore')

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

def calc_rh(q, T, p):
 
    es = 6.112 * np.exp( (17.62 * (T - 273.15)) / (243.12 - (T - 273.15)))
    s = 0.622 * es / p
    #s[s==0] = 1e-6
    rh = q / s
    rh[rh<0] = 0
    rh[rh>1] = 1
    return rh

def calc_z(p0, p, T):
    
    return (((p0/p)**(1/5.257) - 1) * T) / 0.0065

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
        q = g['q'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values #-\
                #g['q'][tidx].mean(axis=(-2,-1)).values.reshape(-1,1,1)
    return q, lats, lons, np.array([time[0]])

def temp(files, loc, **kwargs):
    '''Get temperature of a location'''
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
        t = g['temp'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values - 0#\
                #g['temp'][tidx].mean(axis=(-2,-1)).values.reshape(-1,1,1)
    return t, lats, lons, np.array([time[0]])







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

def moist_int(files, loc, **kwargs):
    '''Moisture flux vol integral'''

    qf = xr.open_dataset(files['vert_cent'])
    vf = xr.open_dataset(files['vert_wind'])
    sf = xr.open_dataset(files['surf'])
    tidx, loci, locj = loc
    w = vf['dz_dt'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    v = vf['v'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    u = vf['u'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    q = qf['q'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    T = qf['temp'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    sp = sf['p'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    P = vf['p'][:].values.reshape(-1, 1, 1) * 100.
    try:
        lons = vf['lon'][locj[0]:locj[1]].values
        lats = vf['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = vf['longitude'][locj[0]:locj[1]].values
        lats = vf['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(vf['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    time = np.array([time[0]])
    Z = calc_z(sp, P, T).mean(axis=(-2, -1))
    dy = cdist((lons[0], lats[0]), (lons[0], lats[1]))*1000
    dx = cdist((lons[0], lats[0]), (lons[1], lats[0]))*1000
    dqvdy = np.gradient(v, Z, dy, dx)[1]
    dqudx = np.gradient(u, Z, dy, dx)[2]
    dqwdz = np.gradient(w, Z, dy, dx)[0]

    div = dqwdz + dqvdy + dqudx

    vf.close(), qf.close(), sf.close()

    return div, lats, lons, np.array([time[0]])


def pbl_t(files, loc, **kwargs):
    '''Get boundary layer properties'''
    f = xr.open_dataset(files['rain'])
    g = xr.open_dataset(files['vera'])
    tidx, loci, locj = loc #Index
    time_1 = pd.DatetimeIndex(f['t'][:].values)
    time_2 = pd.DatetimeIndex(g['t'][:].values)
    tid = np.fabs((time_2 - time_1[tidx]).total_seconds()).argmin()
    f.close()
    return tid , g

def ustar(files, loc, **kwargs):
    '''Get the fricional velocity'''
    _, loci, locj = loc #Index
    tidx, f = pbl_t(files, loc, **kwargs)
    ustar = f['field1696'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')

    f.close()
    return ustar, lats, lons, np.array([time[0]])

def bowen(files, loc, **kwargs):
    '''Get the latent heatflux'''
    _, loci, locj = loc #Index
    tidx, f = pbl_t(files, loc, **kwargs)
    lh = f['lh'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    sh = f['sh'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')

    f.close()
    return sh/(lh+0.001), lats, lons, np.array([time[0]])

def pbl_type(files, loc, **kwargs):
    '''Get the latent heatflux'''
    _, loci, locj = loc #Index
    tidx, f = pbl_t(files, loc, **kwargs)
    diag_type = f['field1036'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')

    f.close()
    return diag_type, lats, lons, np.array([time[0]])



def pbl_h(files, loc, **kwargs):
    '''Get the boundary layer height'''
    _, loci, locj = loc #Index
    tidx, f = pbl_t(files, loc, **kwargs)
    ht = f['field1534'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')

    f.close()
    return ht, lats, lons, np.array([time[0]])

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

def cloud_pf(files, loc, **kwargs):
    from model2lev import interp
    f, g = xr.open_dataset(files['qcl_th']), xr.open_dataset(files['qcf_th'])
    p = xr.open_dataset(files['res_th'])
    Pf = xr.open_dataset(files['vert_cent'])
    P = Pf.variables['p'].values
    Pf.close()

    tidx, loci, locj = loc #Index
    qcf = g['QCF'][tidx-2:tidx].mean(axis=0)
    pres = p['p'][tidx-2:tidx].mean(axis=0)
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    try:
        data = np.ma.masked_invalid(qcf.values).filled(0)
    except AttributeError:
        data = qcf.values
    cloud = interp(data, pres.values/100, P)
    cloudp = cloud[...,loci[0]:loci[1], locj[0]:locj[-1]] - cloud.mean(axis=(-2,-1)).reshape(-1,1,1)

    f.close(), g.close(), p.close()
    return 1000 * cloudp, lats, lons, np.array([time[0]])

def cloud_pl(files, loc, **kwargs):
    from model2lev import interp
    f, g = xr.open_dataset(files['qcl_th']), xr.open_dataset(files['qcf_th'])
    p = xr.open_dataset(files['res_th'])
    Pf = xr.open_dataset(files['vert_cent'])
    P = Pf.variables['p'].values
    Pf.close()

    tidx, loci, locj = loc #Index
    qcl = f['QCL'][tidx-2:tidx].mean(axis=0)
    pres = p['p'][tidx-2:tidx].mean(axis=0)
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    
    try:
        data = np.ma.masked_invalid(qcl.values).filled(0)
    except AttributeError:
        data = qcl.values
    
    cloud = interp(data, pres.values/100, P)
    cloudp = cloud[...,loci[0]:loci[1], locj[0]:locj[-1]] - cloud.mean(axis=(-2,-1)).reshape(-1,1,1)

    f.close(), g.close(), p.close()
    return 1000 * cloudp, lats, lons, np.array([time[0]])




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

def z_lev(files, loc, **kwargs):
    '''Get the z level'''

    v = xr.open_dataset(files['vert_cent'])
    z = xr.open_dataset(files['surf'])

    P = v['p'][:].values * 100

    tidx, loci, locj = loc #Index
    sp = z['p'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    T = v['temp'][tidx, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    Z = calc_z(sp, P.reshape(-1,1,1), T)

    try:
        lons = v['lon'][locj[0]:locj[1]].values
        lats = v['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = v['longitude'][locj[0]:locj[1]].values
        lats = v['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(v['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    v.close(), z.close()

    return Z, lats, lons, np.array(time[0])

def msep(files, loc, **kwargs):
    '''Calculate entrainment rate'''
    from model2lev import interp

    cpd = 1005.7
    Lv = 2.51e6
    Lf = 2.83e6
    ff = xr.open_dataset(files['qcl_th']) 
    gf = xr.open_dataset(files['qcf_th'])
    hf = xr.open_dataset(files['surf'])
    fi = xr.open_dataset(files['res_th'])
    jf = xr.open_dataset(files['vert_cent'])
    kf = xr.open_dataset(files['vert_wind'])
    P = jf['p'][:].values * 100
    tidx, loci, locj = loc #Index
    try:
        tdelta = int(kwargs['tdelta'])
    except KeyError:
        tdelta = 1



    # Get the time windox from timedelta (tdelta in hours)
    time = pd.DatetimeIndex(jf['t'][:].values)
    start = time[tidx] - timedelta(hours=tdelta)
    end = time[tidx] + timedelta(hours=tdelta)
    ttidx = np.where((time>= start)  & (time<= end))[0]
    t1, t2 = ttidx[0], ttidx[-1]+1
    mt = (t2 - t1) // 2


    v = kf['v'][t1:t2, :, loci[0]:loci[1], locj[0]:locj[1]] - kf['v'][t1:t2].mean(axis=(-2, -1))
    u = kf['u'][t1:t2, :, loci[0]:loci[1], locj[0]:locj[1]] - kf['u'][t1:t2].mean(axis=(-2, -1))


    #qcf = g['QCF'][tidx].values#, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    #QF = interp(qcf, p, P[:])
    p = hf['p'][t1:t2].values#, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    T = jf['temp'][t1:t2].values#, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    q = jf['q'][t1:t2].values#, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    Z = calc_z(p, P.reshape(-1,1,1), T)
    H = cpd*T + 9.81*Z + Lv*q #- Lf*QF
    h = H[...,loci[0]:loci[1], locj[0]:locj[1]]
    h -= np.nanmean(H, axis=(-2, -1)).reshape(-1,len(P), 1, 1)
    
    vhp = (np.mean(u*h, axis=0) - np.mean(h, axis=0) * np.mean(h, axis=0)).values
    uhp = (np.mean(v*h, axis=0) - np.mean(h, axis=0) * np.mean(h, axis=0)).values
    try:
        lons = jf['lon'][locj[0]:locj[1]].values
        lats = jf['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = jf['longitude'][locj[0]:locj[1]].values
        lats = jf['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(jf['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    ff.close(), gf.close(), hf.close(), fi.close(), jf.close(), kf.close()
    return  np.sqrt(uhp**2 + vhp**2), lats, lons, np.array(time[0])

def mse_bar(files, loc, **kwargs):
    '''Calculate entrainment rate'''
    from model2lev import interp

    cpd = 1005.7
    Lv = 2.51e6
    Lf = 2.83e6
    f, g = xr.open_dataset(files['qcl_th']), xr.open_dataset(files['qcf_th'])
    z = xr.open_dataset(files['surf'])
    pf = xr.open_dataset(files['res_th'])
    v = xr.open_dataset(files['vert_cent'])
    P = v['p'][:].values * 100
    tidx, loci, locj = loc #Index
    qcf = g['QCF'][tidx].values#, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    p = pf['p'][tidx].values#, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    QF = interp(qcf, p, P[:])
    T = v['temp'][tidx].values#, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    q = v['q'][tidx].values#, :, loci[0]:loci[1], locj[0]:locj[-1]].values
    Z = calc_z(z['p'][tidx].values, P.reshape(-1,1,1), T)
    H = cpd*T + 9.81*Z + Lv*q - Lf*QF
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    f.close(), g.close(), z.close(), v.close(), pf.close()
    return np.nanmean(H, axis=(-2, -1)).reshape(-1, 1, 1), lats, lons, np.array(time[0])

def z(files, loc, **kwags):
    z = xr.open_dataset(files['surf'])
    tidx, loci, locj = loc #Index
    tidx -= 1
    v = xr.open_dataset(files['vert_cent'])
    P = v['p'][:].values * 100
    T = v['temp'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values
    Z = calc_z(z['p'][tidx, :, loci[0]:loci[1], locj[0]:locj[1]].values,
               P.reshape(-1,1,1), T)
    try:
        lons = v['lon'][locj[0]:locj[1]].values
        lats = v['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = v['longitude'][locj[0]:locj[1]].values
        lats = v['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(v['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    v.close(), z.close()
    return Z, lats, lons, np.array(time[0])

def mse(files, loc, **kwargs):
    '''Calculate entrainment rate'''
    from model2lev import interp
    try:
        perturb = kwargs['perturb'].lower() == 'true'
    except:
        perturb = True
    cpd = 1005.7
    Lv = 2.51e6
    Lf = 2.83e6
    f, g = xr.open_dataset(files['qcl_th']), xr.open_dataset(files['qcf_th'])
    z = xr.open_dataset(files['surf'])
    pf = xr.open_dataset(files['res_th'])
    v = xr.open_dataset(files['vert_cent'])
    P = v['p'][:].values * 100
    tidx, loci, locj = loc #Index
    tidx -= 1
    try:
        qci = np.ma.masked_invalid(g['QCF'][tidx].values).filled(0)
    except:
        qci = g['QCF'][tidx].values
    try:
        qcl = np.ma.masked_invalid(f['QCL'][tidx].values).filled(0)
    except:
        qcl = f['QCL'][tidx].values


    p = pf['p'][tidx].values
    QI = interp(qci, p, P[:])
    QL = interp(qcl, p, P[:])
    QI[QI < 0] = 0
    QL[QL < 0] = 0
    T = v['temp'][tidx].values
    q = v['q'][tidx].values
    Z = calc_z(z['p'][tidx].values, P.reshape(-1,1,1), T)
    Tv = T*(1+0.61*q-QL)

    sli = cpd*Tv + 9.81*Z - Lv*(QL) #- Lf*QI
    qt = (q+QL)
    sv = sli - 0.61*cpd*T[0]*qt
    if perturb:
        sv = sv[:,loci[0]:loci[1],locj[0]:locj[1]] -  sv.mean(axis=(-2,-1)).reshape(-1, 1, 1)
    else:
        sv = sv[:,loci[0]:loci[1],locj[0]:locj[1]]
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    f.close(), g.close(), z.close(), v.close(), pf.close()
    return sv/cpd, lats, lons, np.array(time[0])


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

def wet_bulb(files, loc, **kwargs):
    try:
        tdelta = int(kwargs['tdelta'])
    except KeyError:
        tdelta = 1

    f = xr.open_dataset(files['vert_cent'])
    tidx, loci, locj = loc #Index
    tidx -= 2
    #Get the time windox from timedelta (tdelta in hours)
    time = pd.DatetimeIndex(f['t'][:].values)

    P = f.variables['p'][:].values
    q = f.variables['q'][tidx].values#, :, loci[0]:loci[1], locj[0]:locj[1]].values
    T = f.variables['temp'][tidx].values#, :, loci[0]:loci[1], locj[0]:locj[1]].values
    rh = calc_rh(q, T, P.reshape(-1, 1, 1)) * 100
    T = T - 273.15
    wetb = T - (-5.806+0.672*T-0.006*T*T+(0.061+0.004*T+0.000099*T*T)*rh+(-0.000033-0.000005*T-0.0000001*T*T)*rh*rh)
    wetb = wetb[:,loci[0]:loci[1],locj[0]:locj[1]] - wetb.mean(axis=(-2, -1)).reshape(-1, 1, 1)
    try:
        lons = f['lon'][locj[0]:locj[1]].values
        lats = f['lat'][loci[0]:loci[1]].values
    except KeyError:
        lons = f['longitude'][locj[0]:locj[1]].values
        lats = f['latitude'][loci[0]:loci[1]].values

    time = date2num(pd.DatetimeIndex(f['t'][tidx:tidx+1].values).to_pydatetime(),
                    'seconds since 1970-01-01 00:00:00')
    f.close()
    return wetb, lats, lons, np.array([time[0]])


