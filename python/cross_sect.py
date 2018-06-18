import numpy as np
from netCDF4 import Dataset as nc, num2date
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.gridspec as gridspec
import matplotlib
import pandas as pd
import pytz

import math

def recalculate_coordinate(val,  _as=None):
  """
    Accepts a coordinate as a tuple (degree, minutes, seconds)
    You can give only one of them (e.g. only minutes as a floating point number) 
    and it will be duly recalculated into degrees, minutes and seconds.
    Return value can be specified as 'deg', 'min' or 'sec'; default return value is 
    a proper coordinate tuple.
  """
  deg,  min,  sec = val
  # pass outstanding values from right to left
  min = (min or 0) + int(sec) / 60
  sec = sec % 60
  deg = (deg or 0) + int(min) / 60
  min = min % 60
  # pass decimal part from left to right
  dfrac,  dint = math.modf(deg)
  min = min + dfrac * 60
  deg = dint
  mfrac,  mint = math.modf(min)
  sec = sec + mfrac * 60
  min = mint
  if _as:
    sec = sec + min * 60 + deg * 3600
    if _as == 'sec': return sec
    if _as == 'min': return sec / 60
    if _as == 'deg': return sec / 3600
  return deg,  min,  sec
      

def points2distance(start,  end):
  """
    Calculate distance (in kilometers) between two points given as (long, latt) pairs
    based on Haversine formula (http://en.wikipedia.org/wiki/Haversine_formula).
    Implementation inspired by JavaScript implementation from 
    http://www.movable-type.co.uk/scripts/latlong.html
    Accepts coordinates as tuples (deg, min, sec), but coordinates can be given 
    which, not accidentally, is the lattitude of Warsaw, Poland.
  """
  start_long = math.radians(start[0])
  start_latt = math.radians(start[1])
  end_long = math.radians(end[0])
  end_latt = math.radians(end[1])
  d_latt = end_latt - start_latt
  d_long = end_long - start_long
  a = math.sin(d_latt/2)**2 + math.cos(start_latt) * math.cos(end_latt) * math.sin(d_long/2)**2
  c = 2 * math.atan2(math.sqrt(a),  math.sqrt(1-a))
  return 6371 * c / 154.67


def dijkstra(V):
    mask = V.mask
    visit_mask = mask.copy() # mask visited cells
    m = np.ones_like(V) * np.inf
    connectivity = [(i,j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (not (i == j == 0))]
    cc = np.unravel_index(V.argmin(), m.shape) # current_cell
    m[cc] = 0
    P = {}  # dictionary of predecessors 
    #while (~visit_mask).sum() > 0:
    for _ in range(V.size):
        #print cc
        neighbors = [tuple(e) for e in np.asarray(cc) - connectivity 
                     if e[0] > 0 and e[1] > 0 and e[0] < V.shape[0] and e[1] < V.shape[1]]
        neighbors = [ e for e in neighbors if not visit_mask[e] ]
        tentative_distance = np.asarray([V[e]-V[cc] for e in neighbors])
        for i,e in enumerate(neighbors):
            d = tentative_distance[i] + m[cc]
            if d < m[e]:
                m[e] = d
                P[e] = cc
        visit_mask[cc] = True
        m_mask = np.ma.masked_array(m, visit_mask)
        cc = np.unravel_index(m_mask.argmin(), m.shape)
    return m, P

def shortestPath(start, end, P):
    Path = []
    step = end
    while 1:
        Path.append(step)
        if step == start: break
        step = P[step]
    Path.reverse()
    return asarray(Path)

def haversine(azi1, alt1, azi2, alt2):
    """
    Calculate the great circle distance between two points 
    """
    # convert decimal degrees to radians 
    azi1, alt1, azi2, alt2 = map(np.deg2rad, [azi1, alt1, azi2, alt2])

    # haversine formula 
    dlon = azi2 - azi1 
    dlat = alt2 - alt1 
    a = np.sin(dlat/2)**2 + np.cos(alt1) * np.cos(alt2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 

    return np.rad2deg(c)

def getCrossSection(start, end, lon, lat, mul):
    ''' The a cross section between two points '''
    slat = np.argmin(np.fabs(start[-1]-lat))
    elat = np.argmin(np.fabs(end[-1]-lat))
    slon = np.argmin(np.fabs(start[0]-lon))
    elon = np.argmin(np.fabs(end[0]-lon))

    m = Basemap(llcrnrlon=lon.min(), urcrnrlon=lon.max(), llcrnrlat=lat.min(),
             urcrnrlat=lat.max(), resolution='c')
    x = np.arange(min(slon,elon), max(slon,elon)+1)
    a = haversine(start[0], end[0], start[1], start[0])
    x1, y1 = m(*start)
    x2, y2 = m(*end)
    th = np.tan(np.fabs(y1-y2)/np.fabs(x1-x2)) * mul
    d = np.zeros([len(lat), len(lon)])
    if end[-1] > start[-1]:
        func = min
        y = start[-1]
    else:
        func = max
        y = end[-1]
    for xx in x[::-1]:
        x1, y1 = m(lon[xx],y)
        x2, y2 =  m(func(start[0], end[0]), lat)
        gk = np.fabs(th * np.fabs(x1-x2) - y2+y1)
        d[np.argmin(gk),xx] = 1

    return np.where(d == 1)
if  __name__ == '__main__':

    f = nc('/home/unimelb.edu.au/mbergemann/Data/test/umnsaa_pd156.nc')
    g = nc('/home/unimelb.edu.au/mbergemann/Data/test/qrparm.orog.nc')
    w0 = f.variables['dz_dt'][-9][:]
    T=pd.DatetimeIndex(num2date(f.variables['t'][:],f.variables['t'].units))\
                       .round('10min').tz_localize('UTC')\
                       .tz_convert(pytz.timezone('Australia/Darwin'))\
                       .tz_localize(None)
    z = f.variables['hybrid_ht'][:]
    lon = f.variables['longitude'][:]
    lat = f.variables['latitude'][:]
    X,Y = np.meshgrid(lon, lat)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    #fig = plt.figure(figsize=(20,15), dpi=72)
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot(gs[0, 0])
    m = Basemap(llcrnrlon=lon.min(), urcrnrlon=lon.max(), llcrnrlat=lat.min(),
             urcrnrlat=lat.max(), resolution='i', ax=ax1)
    end_point = dict(lon=131.339, lat=-11.818)
    start_point = dict(lon=129.995, lat=-11.2285)

    #end_point = dict(lon=131.528, lat=-11.2292)
    #start_point = dict(lon=130.204, lat=-11.9609)
    #ax2 = fig.add_axes([0.27, 0.065, 0.49, 0.1])
    #fig.subplots_adjust(bottom=0.20, hspace=0.0, wspace=0.0)
    d = np.zeros_like(w0[0])
    idx = getCrossSection((start_point['lon'], start_point['lat']),
                           (end_point['lon'], end_point['lat']), lon, lat, 0.905)
    d[idx] = 1
    dist = haversine(start_point['lon'], end_point['lon'],
                     start_point['lat'], end_point['lat'])
    H = g['ht'][0,0,:]
    sec = H[idx]
    H[idx]=-1000
    H = np.ma.masked_equal(H,-1000)
    cmap = matplotlib.cm.terrain
    cmap.set_bad('k')
    im = m.pcolormesh(lon,lat,H, cmap=cmap)
    cbar = m.colorbar(im, pad='1%',size='2%')
    cbar.set_label('Surface height [m]')
    m.scatter(end_point['lon'],end_point['lat'], 25, marker='o', color='k')
    m.scatter(start_point['lon'],start_point['lat'], 55, marker='s',color='k')
    m.drawcoastlines()
    #ax2.fill_between(np.linspace(0,dist,len(sec)), sec.min(), sec, color='k')
    #ax2.set_ylim(0,200)
    #ax2.set_xlabel('Distance from  $\\blacksquare$ [km]')
    #ax2.set_ylabel('Surface Height [m]')
    #fig.savefig('Topo_map.png',dpi=72, bbox_inches='tight')
    #ax2.fill_between(np.linspace(0,dist,len(sec)), sec.min(), sec, color='k'
    #plt.show()








