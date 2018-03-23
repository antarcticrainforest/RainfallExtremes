import os, sys, cv2
import numpy as np
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter as bl
from netCDF4 import Dataset as nc, num2date

class ConvOrganisation(object):
    '''
    Class that calculates convective organisation potential a la White et al. 2018
    (https://doi.org/10.1175/JAS-D-16-0307.1)
    '''

    def __init__(self, filen, varn, groupn=None):
        '''
        Constructor,
        Arguments:
            filen (nc-object) : name of the netcdf file
            varn (str-object) : the variable name
        Keywords:
            goupn (default None) : name of the netcdf-group if data is saved in
                                   groups
        '''
        self.filen = filen
        self.nc = nc(filen)
        #Get the netcdf-data variable
        if isinstance(groupn, type(None)):
            self.ncvar = self.nc.variables[varn]
        else:
            self.ncvar = self.nc.groups[groupn].variables[varn]
        tname = self.__gettime()
        if tname in self.nc.variables:
            self.time = num2date(self.nc.variables[tname][:], self.variables[tname].units)
        else:
            self.time = num2date(self.nc.groups[groupn].variables[tname][:],
                                 self.nc.groups[groupn].variables[tname].units)
        self.contours = []
        self.circles = []
        self.areaPP = self.get_area()**2
    def create_potential(self, ntstep, **kwargs):
        '''
        Method that call all other methods involved in calculating the convective
        potential
        Arguments:
            ntstep (int) : The number of the current timestep
        Keywords:
            kwargs to be passed to other methods
        '''
        sys.stdout.write('\r%s'%(self.time[ntstep].strftime('Reading data for %D %R ... ')))
        sys.stdout.flush()

        #Now get the contours
        self.contours = []
        self.circles = []
        self.get_contours(self.ncvar[ntstep], **kwargs)
        if len(self.circles) < 1:
            return np.nan, np.nan
        elif len(self.circles) < 2:
            return np.nan, self.circles[0][-1]
        else:
            potential = np.zeros([len(self.circles),len(self.circles)]) - 1
            for i in range(len(self.circles)-1):
                c1, a1 = self.circles[i]
                for j in range(i+1,len(self.circles)):
                    c2, a2 = self.circles[j]
                    potential[i,j] = (a1+a2) / (np.linalg.norm(c1-c2))
            pot = 2 * np.ma.masked_equal(potential,0).sum() /( len(self.circles) * ( len(self.circles) - 1 ))
            area = np.array(self.circles)[:,-1]
            return pot, np.array(self.circles)[:,-1].mean()




    def get_area(self):
        lat = list(self.nc.dimensions.keys())[-2]
        lon = list(self.nc.dimensions.keys())[-1]

        lam1 = np.pi/180. * self.nc.variables[lon][:]
        phi1 = np.pi/180. * self.nc.variables[lat][:]
        i = 0
        dlam = np.fabs(lam1[1] - lam1[0])
        dphi = phi1[1] - phi1[0]

        R = 6370.9989
        a = np.sin(dphi/2)**2 + ((np.cos(phi1[i+1]))*(np.cos(phi1[i])*np.sin(dlam/2)**2))
        return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

    def get_contours(self, data, thresh=2, kernel_size=3, sigma=0.15, **kwargs):
        '''
        Method for filtering all rainfall areas
        Arguments:
            data (2d-array) : rainfall data field
        Keywords:
            thresh (int)    : threshold that is appied to the data
            kernel_size  = the size of the stucturing element
            sigma = sigma value for the Gaussian filter
        Return:
            list containing all the contours
        '''
        import cv2
        #Try to make masked values 0
        data = np.ma.masked_less(data, thresh)
        try:
            data = np.ma.masked_invalid(data).filled(0)
        except AttributeError:
            pass
        #Create a unit arraiy
        data[data != 0] = 255
        bw = data.astype(np.uint8)
        #Now detect the edges of the black-and-white image
        #But first close small holes with erosion and dilation
        #For this purpose we need a kernel (this case a cross)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,\
                                            (kernel_size-1, kernel_size-1))
        #erosion and dilation
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, (kernel_size + 5,
                                 kernel_size + 5))
        for c in range(3):
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                                     (kernel_size + 5, kernel_size + 5))

        #blur the closed-hole image
        closed = bl(closed, sigma)
        #Api-change in from openCV 2.x to 3.x check for major version
        cv2vers = int(cv2.__version__[0])
        #Now get the contours with canny-edge detection
        out = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #Now just get the number of coastal features
        if cv2vers < 3:
            contours, hierarchy = out
        else:
            _, contours, hierarchy = out

        for nn, contour in enumerate(contours):
            if hierarchy[0][nn][-1] == -1:
                area = cv2.contourArea(contour) * 2.5**2
                if area > 0 and len(contour) > 4:
                    self.contours.append(contour)
                    (x,y),radius = cv2.minEnclosingCircle(contour)
                    self.circles.append((np.array((x, y)), radius * 2.5))
    def __gettime(self):
        for i in self.ncvar.dimensions:
            if 'time' in i.lower():
                return i


    def __enter__(self):
        ''' Return the netcdf-file object '''
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ''' Close the netcdf-file if done or something went wrong '''
        self.nc.close()

def create_nc(outf, varn, array, lat, lon, time):
    if os.path.isfile(outf):
        mode = 'a'
    else :
        mode = 'w'
    with nc(outf, mode) as netc:
        for name, data in (('lat',lat), ('lon',lon), ('time',time)):
            try:
                netc.createDimension(name,len(data))
                netc.createVariable(name,'f',(name,))
                netc.variables[name][:] = data
            except (RuntimeError, ValueError, OSError):
                pass
        try:
            netc.createVariable(varn,'f', ('time','lat','lon'))
        except  (RuntimeError, ValueError, OSError):
            pass
        netc.variables[varn][:] = array



def main(filename, output, varname='rain_rate', group=None):
    '''
    Method that calls the convective aggregation routine
    Arguments:
        filename (str)  : the name of the netcdf-file containing the rainfall data
        output (str)    : filename where the output should be stored
    Keywords arguments:
        varname (str)   : variable name of the rainfall data (default rain_rate)
        group           : group name if the data is stored in groups (default None)
    '''
    with ConvOrganisation(filename, varname, group) as Org:
        data = np.zeros([144,117,117],dtype='uint8')
        pot, nob, size = [], [], []
        user,sys,chuser,chsys,real=os.times()
        t1 = (user+sys) * 60
        for i in range(len(Org.time)):
            p, s = Org.create_potential(i)
            pot.append(p)
            size.append(s)
            nob.append(len(Org.contours))
            '''
            for j in range(len(Org.contours)):
                c ,a = Org.circles[j]

                cv2.circle(data[i],tuple(c.astype('i')),int(np.sqrt(a)/np.pi),3,1)
                cv2.drawContours(data[i], [Org.contours[j]], -1, 1, -1)
            '''
            #if i == 144:
             #   data =- np.ma.masked_equal(data.astype('f'),0)

             #   create_nc(output,'cnt',data,Org.nc.variables['lon'][:],
             #           Org.nc.variables['lat'][:],Org.nc['10min'].variables['time'][:144])
            #    break
        
        user,sys,chuser,chsys,real=os.times()
        t2 = (user+sys) * 60
        df = pd.DataFrame({'cop':pot,'nclu':nob, 'size': size},index=pd.DatetimeIndex(Org.time[:]))
        df.to_hdf(output,'cop')
        print('ok')


if __name__ == '__main__':
    import pandas as pd
    cpol =  os.path.join(os.getenv('HOME'), 'Data', 'Extremes', 'CPOL','CPOL_1998-2017.nc')
    outp =  os.path.join(os.getenv('HOME'), 'Data', 'Extremes', 'CPOL','CPOL-Aggr.hdf5')
    main(cpol,outp,group='10min')
