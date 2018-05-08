# coding: utf-8
'''
This python notebook looks at the relationship between stages of the Monsoon (burst) and Extreme Events
'''

import pandas as pd
import numpy as np
import xarray as xr
from netCDF4 import Dataset as nc, num2date, MFDataset as mc
import os
from matplotlib import pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import cufflinks as cf
cf.set_config_file(offline=False, world_readable=True, theme='ggplot')
sns.set_style("whitegrid")
from DataFrame import *
from datetime import datetime


# Define the path's where data is storde

# In[2]:


largeScaleFile = os.path.join(os.environ['HOME'], 'Data', 'Extremes', 'CPOL', 'CPOL-large-scale_forcing.pkl')
largeScaleNc = os.path.join(os.environ['HOME'], 'Data', 'Extremes', 'CPOL', 'CPOL-large-scale_forcing.nc')
CPOL = os.path.join(os.environ['HOME'], 'Data', 'Extremes', 'CPOL', 'CPOL_1998-2017.nc')
CopF = os.path.join(os.environ['HOME'], 'Data', 'Extremes', 'CPOL', 'CPOL-Aggr.hdf5')
BurstF = os.path.join(os.environ['HOME'], 'Data', 'Extremes', 'CPOL', 'CPOL_datetimes_austmonsoon_burstday_lags.pkl')
PercF = os.path.join(os.environ['HOME'],'Data','Extremes','CPOL','CPOL_Percentiles.hdf5')
RegimeF = os.path.join(os.environ['HOME'],'Data','Extremes','CPOL','Pope_Regimes.pkl')
extremeTS = os.path.join(os.environ['HOME'],'Data','Extremes','CPOL','CPOL-TS_extrme.pkl')
MonsoonCat = os.path.join(os.environ['HOME'],'Data','Extremes','CPOL','CPOL_datetimes_austmonsoon_burst_categories.pkl')


# Check for the extrme TS

# In[3]:


perc = pd.read_hdf(PercF)
create = False
if not os.path.isfile(extremeTS) or create:
    with pd.HDFStore(extremeTS,'w') as h5:
        for gr in ('10min','1h','3h','6h','24h'):
            if gr =='10min':
                thresh = 1
            else:
                thresh = 75
            print('Adding %s'%gr)
            h5['/P%s'%gr] = get_extremeTS(CPOL,gr,perc['P%s'%gr].loc[99], thresh)


# In[10]:


cop = pd.read_hdf(CopF) #Convective Potential
extr1h = pd.read_hdf(extremeTS,'P1h')
cop.loc[cop['cop'] == np.inf,'cop'] = 200.
monsoon10m = pd.read_pickle(BurstF)
monsoon10m = monsoon10m.loc[cop.index[0]:]
pope10m = pd.read_pickle(RegimeF)
pope10m2 = pd.read_pickle(RegimeF)
monsooncat = pd.read_pickle(MonsoonCat)
a = pd.Series(np.zeros(len(monsooncat)),index=monsooncat.index)
b = pd.Series(np.zeros(len(monsooncat)),index=monsooncat.index)
a.loc[monsooncat == 'notburst'] = 1
b.loc[monsooncat != 'notburst'] = 1
monsoon10m['0'] = a
monsoon10m['2'] = b
monsoon10m = monsoon10m.dropna()
su = monsoon10m.iloc[:,:3].sum(axis=1)
monsoon10m.loc[su == 1, '0'] = 0
monsoon1h = monsoon10m.groupby(pd.TimeGrouper('1h')).mean()
priors = monsoon10m.iloc[:,:3].dropna().sum(axis=1)
priors = priors.index[priors.values == 1]
breaks = monsoon10m.iloc[:,3].dropna()
breaks = breaks.loc[breaks == 1].index
bursts = monsoon10m['2'].loc[monsoon10m['2'] == 1].index
#post = burst10m.iloc[:,4:-1].dropna().sum(axis=1)
#post = post.loc[post == 1].index



#R1 = pope10m.loc[pope10m == 1]
p1 = pope10m.loc[priors]
p2 = pope10m.loc[breaks]
p3 = pope10m.loc[bursts]
regimes = {1:'DryEast',2:'DeepWest',3:'East',4:'ShalWest', 5:'MoistEast'}
for i in range(1,6):
    pope10m.loc[pope10m == i] = regimes[i]


# In[11]:


regCount = []
names = []
for reg in range(1,6):
    regCount.append(int(p2.loc[p2==reg].count()/144))
    names.append(regimes[reg])
regCount = pd.Series(regCount,index=names)
#regCount = pd.DataFrame(np.array(regCount).T, columns=regimes)
regCount.iplot(kind='bar', filename='pre-burst_Count',yTitle='Count of Break Days',xTitle='Regime',
                  title='Monsoon Break vs. Rainfall Regime')


# In[12]:


regCount=[]
names =[]
for reg in range(1,6):
    regCount.append(int(p1.loc[p1==reg].count()/144))
    names.append(regimes[reg])
regCount = pd.Series(regCount,index=names)
#regCount = pd.DataFrame(np.array(regCount).T, columns=regimes)
regCount.iplot(kind='bar', filename='burst_Count',yTitle='Count of Days prior to burst',xTitle='Regime',
                  title='Prior to Monsoon Burst vs. Rainfall Regime')


# In[13]:


regCount=[]
names =[]
for reg in range(1,6):
    regCount.append(int(p3.loc[p3==reg].count()/144))
    names.append(regimes[reg])
regCount = pd.Series(regCount,index=names)
#regCount = pd.DataFrame(np.array(regCount).T, columns=regimes)
regCount.iplot(kind='bar', filename='pre-burst_Count',yTitle='Count of Burst Days',xTitle='Regime',
                  title='Monsoon Burst vs. Rainfall Regime')


# ### Monsoon Stuff

# In[17]:


burst_cnt = []
names = ['Break', 'Before Burst', 'Burst']
monsoonTS = monsoon10m.loc[extr1h['time'].values].dropna()
extr_events = monsoonTS.loc[extr1h['time'].loc[extr1h['event'].values == 1]].index
print(np.unique(extr1h['event'].values))
extr_monsoon = monsoonTS.loc[extr_events]
priors1h = monsoonTS.iloc[:,:3].sum(axis=1)
priors1h = priors1h.index[priors1h.values == 1]
breaks1h = monsoonTS.iloc[:,3]
breaks1h = monsoonTS.loc[breaks1h == 1].index
bursts1h = monsoonTS.iloc[:,4:-1].dropna().sum(axis=1)
bursts1h = bursts1h.loc[bursts1h == 1].index

Nb = len(monsoonTS)
Ne = len(extr_events)

priors_ext = extr_monsoon.iloc[:,:3].sum(axis=1)
priors_ext = priors_ext.index[priors_ext.values == 1]
breaks_ext = extr_monsoon.iloc[:,3]
breaks_ext = breaks_ext.loc[breaks_ext == 1].index
bursts_ext = extr_monsoon.iloc[:,4:-1].sum(axis=1)
bursts_ext = bursts_ext.loc[bursts_ext == 1].index
ii = 1
for ext,cat in ((breaks_ext,breaks1h), (priors_ext,priors1h),(bursts_ext,bursts1h)):
    #Pbe = len(ext)/len(extr1h.dropna()) *100.
    Nbe = len(ext)
    #print(Pbe,Ne/Np*100*len(cat)/len(burstp))
    #if ii == 3:
    #burst_cnt.append(Pbe/ (Ne/Np) - 2)
    #else:
    burst_cnt.append(len(ext)/len(cat)* 100)
    ii += 1
#    N = extrreg.loc[extrreg==float(reg)].count()
#    reg_cnt.append(100 * N/M)
#    names.append(regimes[reg])
burst_cnt = pd.Series(burst_cnt,index=names)
burst_cnt.iplot(kind='bar', filename='reg_Count',yTitle='P(Extrme|Stage)',xTitle='Monsoon Stage',
                  title='Extreme Events vs. Monsoon Stage')


# In[18]:


burst_cnt = []
ii = 1
for ext,cat in ((breaks_ext,breaks1h), (priors_ext,priors1h),(bursts_ext,bursts1h)):
    #Pbe = len(ext)/len(extr1h.dropna()) *100.
    Pbe = len(ext)/Ne * 100 / len(cat)
    
    if ii == 3:
        Pbe *= 0.75
    ii+=1
    
    burst_cnt.append(Pbe)
burst_cnt = pd.Series(burst_cnt,index=names)
print(burst_cnt.values.sum())
burst_cnt.iplot(kind='bar', filename='reg_Count',yTitle='Frequency of Extremes (weighted)',xTitle='Monsoon Stage',
                  title='Extreme Events vs. Monsoon Stage')


# In[390]:





# In[20]:


# First for rainfall regimes
cop_burst = []
cop1h = cop.groupby(pd.TimeGrouper('1h')).mean()
cop_burst.append({
            "type": 'violin',
            #'x': np.repeat('PreBurst',len(cop1h.loc[post_ext,'nclu'].values)),
            "y": cop1h.loc[breaks_ext,'nclu'].values,
            "name": 'Break',
            #"side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        })
cop_burst.append({
            "type": 'violin',
            #'x': np.repeat('PreBurst',len(cop1h.loc[post_ext,'nclu'].values)),
            "y": cop1h.loc[priors_ext,'nclu'].values,
            "name": 'PriorBurst',
            #"side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        })
cop_burst.append({
            "type": 'violin',
            #'x': np.repeat('PreBurst',len(cop1h.loc[post_ext,'nclu'].values)),
            "y": cop1h.loc[bursts_ext,'nclu'].values,
            "name": 'Burst',
            #"side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        })

fig_2 = {
    "data": cop_burst,
    "layout" : {
        "title": "Convective Organisation vs Monsoon Stage",
        "yaxis": {
            "zeroline": False,
            "title": "Number of Convective objects [ ]"
            
        }
    }
}


# In[21]:


py.iplot(fig_2, filename='cop_violine',validate=False)


# ### Regime Stuff

# In[22]:


reg_cnt = []
names = []
idx1 = extr1h.loc[extr1h['event'] == 1]['time']
pope1h = pope10m2.groupby(pd.TimeGrouper('1h')).mean()

extrreg = pope1h.loc[idx1]
M = len(idx1)
pope1h = pope10m2.groupby(pd.TimeGrouper('1h')).mean().dropna()
for reg in range(1,6):
    N = extrreg.loc[extrreg==float(reg)].count()
    reg_cnt.append(100 * N/pope1h.loc[pope1h==float(reg)].count())
    
    #reg_cnt.append(100 * N/M) #/ (len(idx1)/len(extr1h)))
    names.append(regimes[reg])
reg_cnt = pd.Series(reg_cnt,index=names)
reg_cnt.iplot(kind='bar', filename='reg_Count',yTitle='P(Extreme|Regime)',xTitle='Regime',
                  title='Extreme Events vs. Rainfall Regime')


# In[23]:


reg_cnt = []
names = []
idx1 = extr1h.loc[extr1h['event'] == 1]['time']
pope1h = pope10m2.groupby(pd.TimeGrouper('1h')).mean()

extrreg = pope1h.loc[idx1]
M = len(idx1)
pope1h = pope10m2.groupby(pd.TimeGrouper('1h')).mean().dropna()
for reg in range(1,6):
    N = extrreg.loc[extrreg==float(reg)].count()
    #reg_cnt.append(100 * N/len(pope1h) / (len(idx1)/len(extr1h)))
    reg_cnt.append(100 * N/M) #/ (len(idx1)/len(extr1h)))
    names.append(regimes[reg])
print(np.array(reg_cnt).sum())
reg_cnt = pd.Series(reg_cnt,index=names)
reg_cnt.iplot(kind='bar', filename='reg_Count',yTitle='Frequency of Event Occurrance',xTitle='Regime',
                  title='Extreme Events vs. Rainfall Regime')


# ### Organization Stuff

# In[24]:


# First for rainfall regimes
cop_reg = []
cop1h = cop.groupby(pd.TimeGrouper('1h')).mean()
idx1 = extr1h.loc[extr1h['event'] == 1]['time']
copextr = cop.loc[idx1].dropna()
copextr['regime'] = pope10m.loc[idx1]
copextr = copextr.dropna()

cop_reg.append({
            "type": 'violin',
            'x': np.repeat('All',len(copextr['size'].values)),
            "y": copextr['nclu'].values,
            "name": 'All',
            #"side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        })
'''
cop_reg.append(
    {
            "type": 'violin',
            'x': np.repeat('All',len(copextr['nclu'].values)),
            "y": copextr['nclu'].values,
            "name": 'Convective Objects',
            'side':'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }

)
'''
for reg in range(1,6):
        cop_reg.append({
            "type": 'violin',
            'x': copextr['regime'].loc[copextr['regime']== regimes[reg]],
            "y": copextr['nclu'].loc[copextr['regime'] == regimes[reg]].values,
            "name": regimes[reg],
           # 'side': 'positive',
            "box": {"visible": True},"meanline": {"visible": True}})
        '''
        cop_reg.append({
            "type": 'violin',
            'x': copextr['regime'].loc[copextr['regime']== regimes[reg]],
            "y": copextr['nclu'].loc[copextr['regime'] == regimes[reg]].values,
            "name": 'Convective Objects',
            'side': 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        })'''
        
        
    
fig_1 = {
    "data": cop_reg,
    "layout" : {
        "title": "Convective Organisation across Different Rainfall Regimes",
        "yaxis": {
            "zeroline": False,
            "title": "Nuber of Convective Objects [ ]"
            
        }
    }
}


# In[25]:


py.iplot(fig_1, filename='cop_violine',validate=False)


# ### Maps of frequency

# In[26]:


from mpl_toolkits.basemap import Basemap


# In[27]:


with nc(CPOL) as fnc:
    lon = fnc.variables['lon'][:]
    lat = fnc.variables['lat'][:]
    time = pd.DatetimeIndex(num2date(fnc['1h'].variables['time'][:],fnc['1h'].variables['time'].units))
    rr = fnc['1h'].variables['rain_rate-flip'][:]


# In[28]:


m = Basemap(llcrnrlat=min(lat), llcrnrlon=min(lon), urcrnrlat=max(lat), urcrnrlon=max(lon), resolution='f')


# In[29]:


with nc(CPOL) as fnc:
    rr10min = fnc['10min'].variables['rain_rate'][2495]
    rr6h = fnc['6h'].variables['rain_rate'][69]
    rr24h = fnc['24h'].variables['rain_rate'][17]


# In[32]:


names = ['10 minute snapshot', '6 hourly avg.', '24 hourly avg.']
fig = plt.figure()
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [30,30]

from mpl_toolkits.basemap import cm
for i, rain in enumerate((rr10min,rr6h,rr24h)):
    ax = fig.add_subplot(1,3,i+1)
    im = m.pcolormesh(lon, lat, rain,vmin=0.2,vmax=5,cmap=cm.GMT_drywet)
    m.drawcoastlines(color='k')
    cbar=m.colorbar(im,location='bottom',pad='5%')
    cbar.set_label('mm/h',size=22)
    cbar.ax.tick_params(labelsize=22)
    ax.set_title(names[i],size=22)


# In[33]:


thresh = perc['P1h'][99]
S = pd.Series(np.arange(len(time)),index=time)
S.loc[time].values


# In[34]:


D = []
data = np.zeros([117,117])
for i in (time, breaks1h, priors1h, bursts1h):
    tt = S.loc[i].values
    D.append((( np.ma.masked_less(rr[:,:,tt],thresh) * 0 ) + 1 ).sum(axis=-1))


# In[35]:


import matplotlib as mpl


# In[38]:


names=['(total)','during breaks','before bursts','during bursts']
mpl.rcParams['figure.figsize'] = [30.0,30.0]
maxval =[180, 80, 20, 80]
C  = [D[0],D[1],D[2],D[3]]
fig = plt.figure()
mask = (D[0] * 0 )+1
for i, idx in enumerate(C[1:]):
    ax = fig.add_subplot(1,3,i+1)
    data = np.zeros([117,117])
    #tt = S.loc[time].values
    #data = (( np.ma.masked_less(rr[:,:,tt],thresh) * 0 ) + 1 ).sum(axis=-1)
    try:
        im = m.pcolormesh(lon, lat, mask*C[i+1].filled(0),vmax=maxval[i+1],cmap=mpl.cm.magma)
    except AttributeError:
        im = m.pcolormesh(lon, lat, mask*C[i+1],vmax=maxval[i+1], cmap=mpl.cm.magma)
    m.drawcoastlines(color='w')
    cbar=m.colorbar(im,location='bottom',pad='5%')
    cbar.set_label(u'N Events above 99th%',size=18)
    cbar.ax.tick_params(labelsize=18) 
    ax.set_title('Extremes %s'%names[i+1],size=18)


# In[39]:


ax = fig.add_subplot(111)
i=0
mpl.rcParams['figure.figsize'] = [15,15]
try:
    im = m.pcolormesh(lon, lat, mask*C[i].filled(0),vmax=maxval[i],cmap=mpl.cm.magma)
except AttributeError:
    im = m.pcolormesh(lon, lat, mask*C[i],vmax=maxval[i],cmap=mpl.cm.magma)
m.drawcoastlines(color='w')
cbar=m.colorbar(im,location='bottom',pad='5%')
cbar.set_label(u'N Events above 99th%',size=22)
cbar.ax.tick_params(labelsize=22) 
ax.set_title('Extremes %s'%names[i])


# In[40]:


breaks = np.unique(breaks1h.date)
priors = np.unique(priors1h.date)
bursts = np.unique(bursts1h.date)
DFFile = os.path.join(os.environ['HOME'],'Data','Extremes','CPOL','CPOL_MonsoonPhases-dates.hdf5')
periods = {}
periods['breaks'] = pd.DataFrame(datetime_range([d for d in breaks if d not in bursts and d not in priors]),columns=['start','end'])
periods['priors'] = pd.DataFrame(datetime_range([d for d in priors if d not in bursts and d not in breaks]),columns=['start','end'])
periods['bursts'] = pd.DataFrame(datetime_range([d for d in bursts if d not in breaks and d not in priors]),columns=['start','end'])


# In[41]:


with pd.HDFStore(DFFile,'w',protocol=2) as h5:
    for key, value in periods.items():
        h5['/%s'%key] = value


# In[42]:


pd.read_hdf(DFFile,'/breaks')

