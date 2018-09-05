from tint import Cell_tracks, animate

import os, pandas as pd
from itertools import groupby
import numpy as np
from netCDF4 import Dataset as nc, num2date, date2num
from datetime import datetime, timedelta
import sys

def spl(present, time):
    out = []
    start = True
    a = ''.join(list(present.astype(int).astype(str)))
    b = list(present.astype(int).astype(str))
    ii = 0
    for k, g in groupby(a):
        gg = list(g)
        if len(gg) == 1:
            b[ii] = str(1 - int(gg[0]))
        ii += len(gg)
    ii = 0
    for k, g in groupby(''.join(b)):
        G = list(g)
        if k == '1':
            kk = 0
            for i in G:
                d1 = time[kk+ii]
                try:
                    d2 = time[kk+ii+1]
                except IndexError:
                    break
                if (d2 - d1) > 24*60**2:
                    break
                kk+=1
            if (ii+kk-1) - ii > 5:
                out.append((ii,ii+kk-1))
        ii+=len(G)
    return out

def get_grids(group, slices, lon, lat, varname='rain_rate', timename='time'):
    x = group.variables[varname].shape[-2]
    y = group.variables[varname].shape[-1]
    print(x,len(lon),y,len(lat))
    for s in range(slices[0], slices[-1]+1):
        try:
            yield {'x': lon, 'y': lat,
                   'data':group.variables[varname][s].reshape(1,x,y),
                   'time': num2date(group.variables[timename][s],
                    group.variables[timename].units)}
        except ValueError:
            yield {'x': lon, 'y': lat,
                   'data':group.variables[varname][s,0].reshape(1,x,y),
                   'time': num2date(group.variables[timename][s],
                    group.variables[timename].units)}
def get_times(start, end, time):
    '''Get the start and end index for a given period'''
    if type(end) == type('a'):
        end = datetime.strptime(end, '%Y-%m-%d %h:%M')
    if type(start) == type('a'):
        start = datetime.strptime(start, '%Y-%m-%d %h:%M')

    start = date2num([start], time.units)
    end = date2num([end], time.units)
    e_idx = np.argmin(np.fabs(time[:] - end))+1
    if e_idx == len(time[:]):
        e_idx -= 1
    s_idx = np.argmin(np.fabs(time[:] - start))

    return [(s_idx,e_idx)]

def creat_tracks(dataF,
                 start=None,
                 end=None,
                 overwrite=True,
                 ani=True,
                 varname='lsrain',
                 timename='t',
                 lonname='lon',
                 latname='lat',
                 group=None,
                 **kwargs):
    '''
    Create tintV2 tracks 

    Arguments : 
        dataF (str) : the netCDF file that contains the rainfall data
    Key word args:
        start : If start is given first time step of the tracking, if None 
                start is the first time step of the data.
                Format should be 'YYYY-mm-dd hh:MM' (default : None)
        end  : If send is given last time step of the tracking, if None 
                end is the last time step of the data (
                Format should be 'YYYY-mm-dd hh:MM' default : None)
        overwrite : If a track already exist, delete it an crate a new one,
                    (dfault : True)
        ani  : Create a movie of the tracked systems
    '''

    trackdir = os.path.join(os.path.dirname(dataF),'Tracking')
    sys.stdout.write('Creating tracks for %s\n'%dataF)
    try:
        os.makedirs(os.path.join(trackdir, 'video'))
    except FileExistsError:
        pass

    #start = '2006-11-10 00:00'
    #end = '2006-11-18 18:00'
    with nc(dataF) as ncf:
        if group is not None:
            g = ncf[group]
        else:
            g = ncf

        if type(start) == type('a') and type(end) == type('a'):
            slices = get_times(start, end, g.variables[timename])
        else:
            try:
                slices = spl(g.variables['ispresent'][:],
                             g.variables[timename][:])
            except KeyError:
                start = num2date(g.variables[timename][:],
                                 g.variables[timename].units)[0]
                end  = num2date(g.variables[timename][:],
                                 g.variables[timename].units)[-1]
                slices = get_times(start, end, g.variables[timename])

        lats = ncf.variables[latname][:]
        lons = ncf.variables[lonname][:]
        if len(lats.shape) == 2:
            lats = lats[:,0]
        if len(lons.shape) == 2:
            lons = lons[0,:]

        x = lons[len(lons)//2]
        y = lats[len(lats)//2]
        grids = []
        for s in slices:
            ani = False
            gr = (i for i in get_grids(g, s, lons, lats, varname=varname,
                                       timename=timename))
            anim = (i for i in get_grids(g, s, lons, lats, timename=timename,
                                         varname=varname))
            start = num2date(g.variables[timename][s[0]],
                             g.variables[timename].units)

            end = num2date(g.variables[timename][s[-1]],
                           g.variables[timename].units)
            suffix = '%s-%s'%(start.strftime('%Y_%m_%d_%H'), end.strftime('%Y_%m_%d_%H'))
            tracks_obj = Cell_tracks()
            tracks_obj.params['MIN_SIZE'] = 4
            tracks_obj.params['FIELD_THRESH'] = 1
            track_file = os.path.join(trackdir,'tint_tracks_%s.pkl'%suffix)
            if not os.path.isfile(track_file) or overwrite:
                ncells = tracks_obj.get_tracks(gr, (x,y))
                if ncells > 2 :
                    tracks_obj.tracks.to_pickle(track_file)
                    ani = True
                else:
                    ani = False
            '''
            else:
                try:
                    tracks_obj.tracks = pd.read_pickle(track_file)
                    tracks_obj.radar_info = {'radar_lat':y, 'radar_lon':x}
                    ani = True
                except FileNotFoundError:
                    ani = False
            '''
            if ani:
                animate(tracks_obj, anim,
                        os.path.join(trackdir,'video', 'tint_tracks_%s.mp4'%suffix),
                        overwrite=overwrite, dt = 9.5, **kwargs)
            #break

def get_mintime(ensembles, Simend):
    ''' Construct the filenames of the UM -
        rainfall ouput and get the overlapping time periods
    '''
    start, end = [], []
    data_files = []
    for ens in ensembles:
        date = datetime.strptime(ens,'%Y%m%dT%H%MZ')
        umf133 = 'um-1p33km-%s-rain_%s-%s.nc' %(date.strftime('%m%d%H%M'), date.strftime('%Y%m%d_%H%M'), Simend)
        umf044 = 'um-0p44km-%s-rain_%s-%s.nc' %(date.strftime('%m%d%H%M'), date.strftime('%Y%m%d_%H%M'), Simend)
        data_files.append((os.path.join(UMdir, ens, 'darwin', '1p33km', umf133),
                           os.path.join(UMdir, ens, 'darwin', '0p44km',umf044)))
        time133 = nc(os.path.join(UMdir,ens,'darwin','1p33km', umf133)).variables['t']
        time044 = nc(os.path.join(UMdir,ens,'darwin','0p44km', umf044)).variables['t']
        start.append((num2date(time133[:],time133.units)[0], num2date(time044[:], time044.units)[-1]))
        end.append((num2date(time133[:],time133.units)[-1], num2date(time044[:], time044.units)[-1]))
    return  data_files,\
            np.max(np.array(start),axis=0).min(),\
            np.min(np.array(end), axis=0).max()

if __name__ == '__main__':
    dataF = os.path.join(os.getenv("HOME"),'Data',
                         'Extremes','CPOL','CPOL_TIWI_1998-2017-old.nc')
    overwrite = True
    WOHFv1 = os.path.join(os.environ['HOME'], 'Data', 'Extremes', 'CPOL',
                          'CPOL_WOHv3.nc')
    UMdir = os.path.join(os.getenv('HOME'), 'Data', 'Extremes', 'UM',
                        'darwin', 'RA1T')
    ensembles = ('20061109T1200Z', '20061109T1800Z', '20061110T0000Z',
                 '20061110T0600Z', '20061110T1200Z', '20061110T1800Z',
                 '20061111T0000Z', '20061111T1200Z')
    remap_res = '2.5km'
    Simend = '20061119_0600-%s'%remap_res
    umfiles, start, end = get_mintime(ensembles, Simend)
    #creat_tracks(WOHFv1, start, end, latname='lat', timename='time',
    #             lonname='lon', keep_frames=True)
    for umf133, umf044 in umfiles:
        for fname in (umf133, umf044):
            creat_tracks(fname, start, end, latname='lat',
                        lonname='lon', keep_frames=True)


