from tint_2 import Cell_tracks, animate

import os, pandas as pd
from itertools import groupby
import numpy as np
from netCDF4 import Dataset as nc, num2date, date2num
from datetime import datetime, timedelta

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

def get_grids(group, slices, lon, lat, varname='rain_rate'):
    x = group.variables[varname].shape[1]
    y = group.variables[varname].shape[2]

    for s in range(slices[0], slices[-1]+1):
        yield {'x': lon, 'y': lat,
               'data':group.variables[varname][s].reshape(1,x,y),
               'time': num2date(group.variables['time'][s],
               group.variables['time'].units)}
def get_times(start, end, time):
    '''Get the start and end index for a given period'''
    if type(end) == type('a'):
        end = datetime.strptime(end, '%Y-%m-%d %H:%M')
    if type(start) == type('a'):
        start = datetime.strptime(start, '%Y-%m-%d %H:%M')

    start = date2num([start], time.units)
    end = date2num([end], time.units)
    e_idx = np.argmin(np.fabs(time[:] - end))+1
    s_idx = np.argmin(np.fabs(time[:] - start))
    
    return [(s_idx,e_idx)]


if __name__ == '__main__':
    dataF = os.path.join(os.getenv("HOME"),'Data','Extremes','CPOL','CPOL_1998-2017.nc')
    trackdir = os.path.join(os.path.dirname(dataF),'Tracking')
    overwrite = True
    start = '2006-11-10 00:00'
    end = '2006-11-18 18:00'
    with nc(dataF) as ncf:
        if type(start) == type('a') and type(end) == type('a'):
            slices = get_times(start, end, ncf['10min'].variables['time'])
        else:
            slices = spl(ncf['10min'].variables['ispresent'][:],
                         ncf['10min'].variables['time'][:])

        lats = ncf.variables['lat'][:]
        lons = ncf.variables['lon'][:]
        x = lons[int(117/2)]
        y = lats[int(117/2)]
        grids = []
        for s in slices:
            ani = False
            gr = (i for i in get_grids(ncf['10min'], s, lons, lats))
            anim = (i for i in get_grids(ncf['10min'], s, lons, lats))
            start = num2date(ncf['10min'].variables['time'][s[0]],
                             ncf['10min'].variables['time'].units)

            end = num2date(ncf['10min'].variables['time'][s[-1]],
                           ncf['10min'].variables['time'].units)
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
                        overwrite=overwrite, dt = 9.5)
            #break

