'''
Python module with functions that does the Ensemble manipulation
'''
from datetime import datetime, timedelta
from glob import glob
import importlib
from itertools import product
import os
import random
import re
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
from h5py import File
from netCDF4 import Dataset as nc
import numpy as np
import pandas as pd
import wrf
import xarray as xr
warnings.filterwarnings('default', category=FutureWarning)


from DataFrame import get_rainIndex
import model2lev
import storm_prop

def test(expID, thread):
  '''
  Make some tests for the multiprocessing environment
  '''

  print('%s is working on %s'%(thread, expID))
  time.sleep(4*random.random())
  return 0 #random.randint(0,1)

def get_cloud_top(expID, thread, **kwargs):
    '''
    Calculate cloud top temp
    '''
    try:
        basedir = os.path.expanduser(kwargs['basedir'])
    except KeyError:
        basedir = os.path.expanduser('~/Data/Extremes/UM/darwin/RA1T')

    expDir = datetime.strptime(expID, 'u-%m%d%H%M').strftime('2006%m%dT%H%MZ')


    lookup = {'UM044': '0p44km' , 'UM133': '1p33km'}
    #Then get the netCDF files
    ncfiles = {'UM044': {}, 'UM133': {}}
    for res in ncfiles.keys():
        for ncf in glob(os.path.join(basedir,'um-%s-%s-*.nc'%(lookup[res],
                                                              expID.strip('u-')))):
            txt = os.path.basename(ncf).strip('um-%s-%s-'%(lookup[res], expID)).split('_')
            v = '_'.join([vv for vv in txt[:2] if not vv.startswith('2')])
            ncfiles[res][v] = ncf
    for res in ncfiles.keys():
        f = xr.open_dataset(ncfiles[res]['qcl_th'])
        ff = xr.open_dataset(ncfiles[res]['geop_th'])
        g = xr.open_dataset(ncfiles[res]['res_th'])
        gg = xr.open_dataset(ncfiles[res]['qcf_th'])
        w = xr.open_dataset(ncfiles[res]['vert_cent'])
        zf = xr.open_dataset(os.path.join(basedir,'orog.nc'))
        
        p = w['p'].values
        pp = np.ones((w['temp'].shape[1], w['temp'].shape[-2], w['temp'].shape[-1]))
        P = pp * p.reshape(-1,1,1)
        T = w['temp']

        orog = zf['ht'][0,0]
        ctt = np.empty((w['temp'].shape[0], w['temp'].shape[-2], w['temp'].shape[-1]))
        ht = ff['ht']
        pres = g['p']
        qlm = f['QCL']
        qlf = gg['QCF']
        q = w['q']
        r = q/(1-q)
        for i in range(len(ctt)):
            geop = model2lev.interp(np.copy(ht[i].values, order='C'),
                                    np.copy(pres[i].values/100., order='C'),
                                    np.copy(p, order='C'))
            ql = model2lev.interp(np.copy(qlm[i].values, order='C'),
                                    np.copy(pres[i].values/100., order='C'),
                                    np.copy(p, order='C'))
            qi = model2lev.interp(np.copy(qlf[i].values, order='C'),
                                  np.copy(pres[i].values/100, order='C'),
                                  np.copy(p, order='C'))
            ctt[i] = wrf.ctt(np.copy(P, order='C'),
                             np.copy(T[i].values, order='C'),
                             np.copy(r[i].values, order='C'),
                             np.copy(ql, order='C'),
                             np.copy(geop, order='C'),
                             np.copy(orog.values, order='C'),
                             qice=np.copy(qi, order='C'),
                             meta=False, missing=-9999.99)
        ff.close(), f.close(), g.close(), gg.close(), w.close(), zf.close()
        with nc(ncfiles[res]['surf'], 'a') as f:
            try:
                f.createVariable('ctt','f',('t','lat','lon'), fill_value=-9999.99)
            except:
                pass
            f.variables['ctt'][:] = ctt
            f.variables['ctt'].long_name='Cloud Top Temperature'
            f.variables['ctt'].short_name='ctt'
            f.variables['ctt'].units='degC'
            #f.variables['ctt'].missing_value=-9999.99
    return 0



def get_storm_prop(expID, thread, **kwargs):
    '''
    Calculate properties
    '''

    try:
        funcn = kwargs['func']
    except KeyError:
        funcn = 'bflux'

    func = getattr(storm_prop, funcn)
    #Get the treck data first
    try:
        basedir = os.path.expanduser(kwargs['basedir'])
    except KeyError:
        basedir = os.path.expanduser('~/Data/Extremes/UM/darwin/RA1T')

    expDir = datetime.strptime(expID, 'u-%m%d%H%M').strftime('2006%m%dT%H%MZ')
    um133_trackf = os.path.join(basedir, expDir, 'darwin', '1p33km', 'Tracking',
                               '*.pkl')

    um044_trackf = os.path.join(basedir, expDir, 'darwin', '0p44km', 'Tracking',
                               '*.pkl')
    
    track_prop ={'UM044': glob(um044_trackf)[0], 'UM133': glob(um133_trackf)[0]}

    lookup = {'UM044': '0p44km' , 'UM133': '1p33km'}
    #Then get the netCDF files
    ncfiles = {'UM044': {}, 'UM133': {}}
    for res in ncfiles.keys():
        for ncf in glob(os.path.join(basedir,'um-%s-%s-*.nc'%(lookup[res],
                                                              expID.strip('u-')))):
            txt = os.path.basename(ncf).strip('um-%s-%s-'%(lookup[res], expID)).split('_')
            v = '_'.join([vv for vv in txt[:2] if not vv.startswith('2')])
            ncfiles[res][v] = ncf

    outF = os.path.join(basedir,'Storm_prop-2006%sZ.hdf5'%expID.strip('u-'))

    with File(outF, 'a') as h5:

        for res, trackf in track_prop.items():

            with xr.open_dataset(ncfiles[res]['vert_cent']) as f:
                time = pd.DatetimeIndex(f['t'].values)
                lon = f['lon'].values
                lat = f['lat'].values
                p = f['p'].values

            tracks = pd.read_pickle(trackf)
            uids = np.array(tracks.index.get_level_values('uid')).astype('i')
            uids = np.unique(uids)
            uids.sort()

            try:
                h5['/p'][:] = p
            except KeyError:
                h5['/p'] = p

            try:
                h5['/%s/2006%sZ/uids'%(res,expID.strip('u-'))][:]=uids
            except KeyError:
                h5['/%s/2006%sZ/uids'%(res,expID.strip('u-'))]=uids
            for ii, uid in enumerate(uids):
                df = tracks.xs(str(uid), level='uid')
                precip = df['mean'].mean()
                out = func(ncfiles[res], get_rainIndex(df, lon, lat, time, 'mean'),
                           **kwargs)
                try:
                    h5['%s/2006%sZ/%04i/%s'%(res,expID.strip('u-'), uid, funcn)][:]=out[0]
                except KeyError:
                    h5['%s/2006%sZ/%04i/%s'%(res,expID.strip('u-'), uid, funcn)]=out[0]
                for tt, fn in enumerate(('lat', 'lon', 'time')):
                    try:
                        h5['%s/2006%sZ/%04i/%s'%(res,expID.strip('u-'), uid, fn)][:]=out[tt+1]
                    except KeyError:
                        h5['%s/2006%sZ/%04i/%s'%(res,expID.strip('u-'), uid, fn)]=out[tt+1]

                    setattr(h5['%s/2006%sZ/%04i/%s'%(res,expID.strip('u-'), uid, fn)],
                            'units', 'seconds since 1970-01-01 00:00:00')
                    try:
                        setattr(h5['%s/2006%sZ/%04i/%s'%(res,expID.strip('u-'), uid, fn)],
                                'offset', int(kwargs['tdelta']))
                    except KeyError:
                        setattr(h5['%s/2006%sZ/%04i/%s'%(res,expID.strip('u-'), uid, fn)],
                                'offset', 1)
                    try:
                        h5['%s/2006%sZ/%04i/rain'%(res,expID.strip('u-'), uid)] = precip
                    except RuntimeError:
                        pass

    return 0
def um2nc(expID, thread, **kwargs):
    '''
    Convert all um output to netcdf-file format and rename the files 
    to a meaningful name
    '''
    try:
        res = kwargs['res']
        setup='protoRA1T'
    except KeyError:
        res = '1p33km'
        setup='proto_RA1T'
    try:
       start = datetime.strptime(kwargs['start'], '%Y%m%d%H%M')
    except KeyError:
       start = datetime(1000, 1, 1, 0, 0)
    except ValueError:
       raise RuntimeError('Error date must be of format %Y%m%d%H%M')
    try:
       end = datetime.strptime(kwargs['end'], '%Y%m%d%H%M')
    except KeyError:
       end = datetime(9999, 1, 1, 0, 0)
    except ValueError:
       raise RuntimeError('Error date must be of format %Y%m%d%H%M')
    setup = {'4km':'proto_RA1T', '1p33km':'proto_RA1T', '0p44km':'protoRA1T'}[res]

    try:
        remap = kwargs['remap']
    except KeyError:
        remap = '2.5km'

    try:
        remapFile = os.path.expanduser(kwargs['remapFile'])
    except KeyError:
        remapFile = os.path.join(os.environ['HOME'],'Data',
                                 'CPOL_TIWI_2.5kgrid.txt')

    fileID = dict(ph='rain', pg='geop_th', pf='pres_th', pe='qcl_th',
                  pd='qcf_th', pc='vert_cent', pb='vert_wind', pa='surf')
                  #pvera='vera', pverb='verb', pverc='verc')
    try:
      fileID = {kwargs['fileID']:fileID[kwargs['fileID']]}
    except KeyError:
      fileID = fileID
    
    expID_tmp = expID.replace('u-','u-2006')
    date = datetime.strptime(expID_tmp, 'u-%Y%m%d%H%M')
    umID = date.strftime('%Y%m%dT%H%MZ')

    old_path = os.path.dirname(__file__)
    path = os.path.join(os.getenv('HOME'), 'cylc-run', expID, 'share', 'cycle',
                        umID, 'darwin', res, setup, 'um')
    outpath = os.path.join(os.getenv('HOME'), 'Data', 'UM_1', 'RA1T', umID,
                           'darwin', res)
    if not os.path.isdir(outpath):
      if os.system('mkdir -p %s'%outpath) != 0:
        print('Making dir %s failed'%outpath)
        return 1
    if not os.path.isdir(path):
        print('%s : path does not exist'%path)
        return 1
    os.chdir(outpath)
    def exec(thread, cmd):
      '''Execute command an leave'''
      sys.stdout.write('%s: Executing %s \n'%(thread, cmd))
      out = os.system(cmd)
      if out != 0:
        sys.stdout.write('%s: Error executing %s \n'%(thread, cmd))

      return out

    for umid, ncid in fileID.items():
        outdates = []
        umfiles = [ f for f in glob(os.path.join(path,'umnsaa_%s*'%umid)) if '.nc' not in f]
        umfiles.sort()
        if len(umfiles) == 0:
          sys.stdout.write('No files are excisting in %s\n'%path)
          return 1
        merge = True
        for umfile in umfiles:
            dt = int(re.findall(r'\d+', umfile)[-1])
            if date+timedelta(hours=dt) < start:
                  continue
            if date+timedelta(hours=dt) > end:
                  break
            testfile = glob('um-%s-%s-%s_????????_????-????????_????-%s.nc'\
            %(res, expID.replace('u-',''), ncid, remap))
            if not len(testfile):
                merge = True
                cmd = 'cp %s .'%umfile
                if exec(thread, cmd) != 0:
                  return 1
                cmd = 'um2cdf %s'%(os.path.basename(umfile))
                exec(thread, cmd)
                cmd = 'rm %s'%(os.path.basename(umfile))
                exec(thread, cmd)
                outdate = (date + timedelta(hours=dt)).strftime('%Y%m%d_%H%M')
                outfile = 'um-%s-%s-%s_%s-%s.nc'%(res,
                                                  expID.replace('u-',''),
                                                  ncid,
                                                  outdate,
                                                  remap)
                if remapFile is None:
                     cmd = 'cdo sellonlatbox,130.024,131.58,-11.99,-11.083 %s.nc %s'\
                         %(os.path.basename(umfile), outfile)
                else:
                     cmd = 'cdo remapbil,%s %s.nc %s'%(remapFile,
                                                  os.path.basename(umfile),
                                                  outfile)
                outdates.append(outdate)
                if exec(thread, cmd) != 0:
                  return 1
                cmd = 'rm %s.nc'%(os.path.basename(umfile))
                if exec(thread, cmd) != 0:
                  return 1
            else:
                merge = False

        outdates.sort()
        if merge:
            mergefile = 'um-%s-%s-%s_%s-%s-%s.nc'%(res,
                                                   expID.replace('u-',''),
                                                   ncid,
                                                   outdates[0],
                                                   outdates[-1],
                                                   remap)
            cdofiles = 'um-%s-%s-%s_'%(res, expID.replace('u-',''), ncid)
            cmd = 'cdo mergetime %s* %s'%(cdofiles, mergefile)
            if exec(thread, cmd) != 0:
              return 1

            cmd = 'rm %s????????_????-%s.nc'%(cdofiles, remap)
            if exec(thread, cmd) != 0:
              return 1
    os.chdir(old_path)
    return 0



