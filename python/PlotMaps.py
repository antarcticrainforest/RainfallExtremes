
# coding: utf-8

# ## Notbook that plots a few (fancy) maps in 

# In[1]:


from netCDF4 import Dataset as nc, num2date
import numpy as np
import pandas as pd
import sys, os
import matplotlib
from datetime import timedelta
from matplotlib import pyplot as plt, cm
from mpl_toolkits.basemap import Basemap, cm as cm_b
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



def Plot(periods, phase, cpolF, cmorphF, outDir, rank, itt, ittm):
    print('%2i (%i/%i) : Working on %s'%(rank+1, itt, ittm, phase))
    start, end = list(periods.start.values), list(periods.end.values)
    T =[]
    for s, e in zip(start, end):
        T += list(pd.date_range(s, e, freq='10min').to_pydatetime())
    T = pd.DatetimeIndex(T)
    td = timedelta(minutes = 570)
    P = dict(breaks='During Break Phase', bursts='During Burst Phase', priors='<2 Days Before Burst Phase')[phase]
    #T = T[44:120]
    # Get all timesteps

    # ### Plot the CPOL vs CMORPH map
    # Create the maps first, for this get the metadata

    # In[111]:


    with nc(cmorphF) as fnc:
        lon_cm = fnc.variables['lon'][:]
        lat_cm = fnc.variables['lat'][:]
        t_cm = pd.DatetimeIndex(num2date(fnc.variables['time'][:], fnc.variables['time'].units))
        with nc(cpolF) as gnc:
            lon_cp = gnc.variables['lon'][:]
            lat_cp = gnc.variables['lat'][:]
            cpol_times ={}
            for tt in ('10min', '1h', '3h', '6h'):
                cpol_times[tt] = pd.DatetimeIndex(num2date(gnc[tt].variables['time'][:], gnc[tt].variables['time'].units))
            #tt = np.array([(np.where(t_cm == p)[0][0], np.where(t_cp == p)[0][0])for p in T if p in t_cp and p in t_cm])
            idx = []
            for tt in T:
                if tt in cpol_times['10min']:
                    try:
                        i10m=np.where(cpol_times['10min'] == tt)[0][0]
                        t1h = pd.Timestamp(tt.year, tt.month, tt.day, tt.hour)
                        i1h = np.where(cpol_times['1h'] == t1h)[0][0]
                        c = np.array(3*list(range(int(24/3))));c.sort();c*=3
                        i3h = np.where(cpol_times['3h'] == pd.Timestamp(tt.year, tt.month, tt.day, c[tt.hour]))[0][0]
                        i3hc = np.where(t_cm == pd.Timestamp(tt.year, tt.month, tt.day, c[tt.hour]))[0][0]
                        c = np.array(6*list(range(int(24/6))));c*=6;c.sort()
                        i6h = np.where(cpol_times['6h'] == pd.Timestamp(tt.year, tt.month, tt.day, c[tt.hour]))[0][0]
                        idx.append((i10m,i1h,i3h,i6h,i3hc))
                    except IndexError:
                        pass
            
            idx = np.array(idx)
            rr_cp = {}
            for n,tt in enumerate(('10min', '1h', '3h', '6h')):
                rr_cp[tt] = gnc[tt].variables['rain_rate'][idx[:,n]]
        rr_cm = fnc.variables['precip'][idx[:,-1]]


    # In[155]:


    m_cm = Basemap(llcrnrlat=min(lat_cm), llcrnrlon=min(lon_cm), urcrnrlat=max(lat_cm), urcrnrlon=max(lon_cm), resolution='i')
    matplotlib.rcParams['figure.figsize'] = [30,30]
    avg_type = {'10min':'(10 Min.)', '1h': '(1 hly)', '3h': '(3 hly)', '6h': '(6 hly)'}




    first = True
    for i in range(len(rr_cm)):
        fname=os.path.join(outDir,'comparison-%s-%s.png'%(phase,(cpol_times['10min'][idx[i,0]]+td).strftime('%Y-%m-%d_%H%M')))
        if not os.path.isfile(fname):
            print('%2i (%i/%i): Plotting data for %s (%s)'  %(rank+1, itt, ittm, (cpol_times['10min'][idx[i,0]]+td).strftime('%Y-%m-%d %H:%M'),phase))
            try:
                rr_out = rr_cm[i].filled(0) / 3. /2
            except AttributeError:
                rr_out = rr_cm[i] / 3. /2       

            if first:
                fig = plt.figure(figsize=(20,15), dpi=72)
                fig.suptitle('Rainfall %s'%P, fontsize=26, fontweight='bold')
                ax1 = plt.subplot2grid((3,3), (0,1))
                ax10 = plt.subplot2grid((3,3), (0,0))
                ax6 = plt.subplot2grid((3,3), (0,2))
                ax = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=3)
                m_cm.drawcoastlines()
            outer_title = 'CMORPH rain-rates at %s' %(t_cm[idx[i,-1]]+td).strftime('%Y-%m-%d %H:%M')
            inner_title = 'Cpol rain-rate at %s' %(cpol_times['10min'][idx[i,1]]+td).strftime('%Y-%m-%d %H:%M')
            if first:
                rain_im = {}
                rr_a = {}
                rain_im['3h_cm'] = m_cm.pcolormesh(lon_cm, lat_cm, rr_out, vmin=0.1, vmax=5., cmap='Blues', shading='gouraud')
                cbar=m_cm.colorbar(rain_im['3h_cm'],location='bottom',pad='5%')
                cbar.set_label('Rain-rate [mm/h]',size=24)
                cbar.ax.tick_params(labelsize=26)
                axins = zoomed_inset_axes(ax, 6, loc=3)
                axins.set_xlim(min(lon_cp), max(lon_cp))
                axins.set_ylim(min(lat_cp), max(lon_cp))
                m_cp_i = Basemap(llcrnrlat=min(lat_cp), llcrnrlon=min(lon_cp), urcrnrlat=max(lat_cp), urcrnrlon=max(lon_cp), ax=axins, resolution='f')
                m_cp_1 = Basemap(llcrnrlat=min(lat_cp), llcrnrlon=min(lon_cp), urcrnrlat=max(lat_cp), urcrnrlon=max(lon_cp), ax=ax1, resolution='f')
                m_cp_10 = Basemap(llcrnrlat=min(lat_cp), llcrnrlon=min(lon_cp), urcrnrlat=max(lat_cp), urcrnrlon=max(lon_cp), ax=ax10, resolution='f')
                m_cp_6 = Basemap(llcrnrlat=min(lat_cp), llcrnrlon=min(lon_cp), urcrnrlat=max(lat_cp), urcrnrlon=max(lon_cp), ax=ax6, resolution='f')
                plt.xticks(visible=False)
                plt.yticks(visible=False)
                for m, name, a, ii in ((m_cp_10,'10min', ax10, 0),(m_cp_1,'1h', ax1, 1),(m_cp_i,'3h', axins, 2),(m_cp_6,'6h', ax6, 3)):
                    rain_im[name] = m.pcolormesh(lon_cp, lat_cp, rr_cp[name][i].filled(np.nan), vmin=0.1, vmax=5. ,cmap='Blues', shading='gouraud')
                    m.drawcoastlines()
                    rr_a[name] = a
                    if not name.startswith('3h'):
                        rr_a[name].set_title('Cpol at %s %s'%((cpol_times[name][idx[i,ii]]+td).strftime('%Y-%m-%d %H:%M'), avg_type[name]), size=18)
            else:
                for ii, name in enumerate(('10min','1h','3h','6h','3h_cm')):
                    if name == '3h_cm':
                        rain_im[name].set_array(rr_out.ravel())
                    else:
                        rain_im[name].set_array(rr_cp[name][i].filled(np.nan).ravel())
                    if not name.startswith('3h'):
                        rr_a[name].set_title('Cpol at %s %s'%((cpol_times[name][idx[i,ii]]+td).strftime('%Y-%m-%d %H:%M'), avg_type[name]), size=18)
            ax.set_title(outer_title,size=26)
            if first:
                mark_inset(ax, axins, loc1=4, loc2=1, fc="none", ec="0.5")
                fig.subplots_adjust(top=0.88,wspace=0.01)
            fig.savefig(fname, dpi=72, facecolor='w', format='png', edgecolor='w', bbox_inches='tight')
            first = False
if __name__ == '__main__':
    from mpi4py import MPI
    DFFile = os.path.join(os.environ['HOME'],'Data','Extremes','CPOL','CPOL_MonsoonPhases-dates.hdf5')
    cpolF = os.path.join(os.environ['HOME'], 'Data', 'Extremes', 'CPOL', 'CPOL_1998-2017.nc')
    cmorphF = os.path.join(os.environ['HOME'], 'Data', 'Darwin', 'netcdf', 'Cmorph_1998-2010.nc')
    outDir = os.path.join(os.environ['HOME'], 'Data','Extremes' ,'CPOL', 'Plot')
    #cat = {'bursts':0,'breaks':0,'priors':0}
    #dates = {'bursts':[],'breaks':[],'priors':[]}
    cat = {'bursts':0,'breaks':0}
    dates = {'bursts':[],'breaks':[]}
    split = {}
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.size
    name = comm.name
    with pd.HDFStore(DFFile,'r') as h5:
        for key in dates.keys():
            data = h5["/"+key]
            for i in range(len(data)):
                cat[key]+=1.
                dates[key].append(list(data.iloc[i].values))
    if nproc == 1:
        for key in dates.keys():
            for idxn, idx in enumerate(np.array_split(dates[key],int(len(data)/4.))):
                Plot( pd.DataFrame(idx, columns=['start','end']), key, cpolF, cmorphF, outDir, rank, idxn+1, int(len(data)/4.) )
    elif rank == 0:
        if nproc == 2:
            nproc += 1
        tmp = -1
        for i in cat.keys():
            d = int(round(float(cat[i])/ np.array([v for v in cat.values()]).sum() * nproc,0))
            if tmp < d:
                maxcat = i
                tmp = d
            split[i] = d
        S = np.array([v for v in split.values()]).sum()
        while S > nproc:
            split[maxcat] -= 1
            S = np.array([v for v in split.values()]).sum()
        while S < nproc:
            split[maxcat] += 1
            S = np.array([v for v in split.values()]).sum()
        n = 0
        for k in split.keys():
            split[k] = np.arange(n,split[k]+n)
            n=split[k][-1]+1
        out = {}
        i = 0
        while i < nproc:
            for k in split.keys():
                if i in split[k]:
                    dd = np.array_split(dates[k],len(split[k]))
                    for ii in dd:
                        out[i]=(k,ii)
                        i+=1
    else:
        out = None
    if nproc > 1:
        out = comm.bcast(out, root=0)
        for idxn, idx in enumerate(np.array_split(out[rank][-1],int(len(out[rank][-1])/4.))):
        #    print(rank, out[rank][0], pd.DataFrame(idx, columns=['start','end']))
            Plot( pd.DataFrame(idx, columns=['start','end']), out[rank][0], cpolF, cmorphF, outDir, rank, idxn+1, int(len(out[rank][-1])/4.) )

