#include "radar_error.h"
/* ==== Set up the methods table ====================== */
static PyMethodDef _radarMethods[] = {
   {"err", err, METH_VARARGS},
   {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void init_radar()  {
   (void) Py_InitModule("_radar", _radarMethods);
   import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject *err(PyObject *self, PyObject *args)  {
   PyArrayObject *vecin, *vecout, *latin, *lonin;
  const char *fname;
   double *cin, *cout, dfactor;   // The C vectors to be created to point to the 
   double *lat, *lon;             //   python vectors, cin and cout point to the row
                                  //   of vecin and vecout, respectively
   int ii,i,j,n,m, dims[2];
   /* Parse tuples separately since args will differ between C fcns */
   if (!PyArg_ParseTuple(args, "sO!O!O!d", &fname,
      &PyArray_Type, &vecin, &PyArray_Type, &latin, &PyArray_Type, &lonin, &dfactor))  return NULL;
   if (NULL == vecin)  return NULL;
   if (NULL == lonin)  return NULL;
   if (NULL == latin)  return NULL;

   /* Check that object input is 'double' type and a vector
      Not needed if python wrapper function checks before call to this routine */
   if (not_doublevector(vecin)) return NULL;

   /* Get the dimension of the input */
   n                          = dims[0]=vecin->dimensions[0];
   i                          = latin->dimensions[0];
   j                          = lonin->dimensions[0];
   /* Make a new double vector of same dimension */
   vecout = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_DOUBLE);
   struct errorstats   error_statistics; /* Error statistics at rain gauge pixels. */
   struct radar        radar_data;       /* Holds the radar rain rate data. */
   error_statistics.nstns    = 0;
   error_statistics.lon      = NULL;
   error_statistics.lat      = NULL;
   error_statistics.mu       = NULL;
   error_statistics.sigma    = NULL;

   radar_data.nx             = 0;
   radar_data.ny             = 0;
   radar_data.rain_rate      = NULL;
   radar_data.lon            = NULL;
   radar_data.lat            = NULL;
   radar_data.nearest_stn    = NULL;
   radar_data.rain_rate_pdf  = NULL;

   /* Change contiguous arrays into C *arrays   */
   cin      = pyvector_to_Carrayptrs(vecin);
   lat      = pyvector_to_Carrayptrs(latin);
   lon      = pyvector_to_Carrayptrs(lonin);

   cout                      = pyvector_to_Carrayptrs(vecout);
   //radar_data.rain_rate      = cin;
   //radar_data.lat            = lat;
   //radar_data.lon            = lon;
   radar_data.nx             = j;
   radar_data.ny             = i;

   if (!(radar_data.rain_rate = malloc(radar_data.nx * radar_data.ny * sizeof(double)))){
      fprintf(stderr,"Memory allocation error for rain_rate.\n");
      return NULL;
   }
   if (!(radar_data.lon = malloc(radar_data.nx * sizeof(double)))){
      fprintf(stderr,"Memory allocation error for nx.\n");
      return NULL;
   }
   if (!(radar_data.lat = malloc(radar_data.ny * sizeof(double)))){
      fprintf(stderr,"Memory allocation error for ny.\n");
      return NULL;
   }
   if (!(radar_data.nearest_stn = malloc(radar_data.nx * radar_data.ny * sizeof(int)))){
      fprintf(stderr,"Memory allocation error for nearest_stn.\n");
      return NULL;
   }
   if (!(radar_data.rain_rate_pdf = malloc(radar_data.nx * radar_data.ny * sizeof(double*)))){
      fprintf(stderr,"Memory allocation error pdf.\n");
      return NULL;
   }
   /* Do the calculation. */
   for ( ii=0; ii<n; ii++)  {
         if (ii < i) radar_data.lat[ii] = lat[ii];
         if (ii < j) radar_data.lon[ii] = lon[ii];
         radar_data.rain_rate[ii]= cin[ii];
   }
   if(read_error_stats(fname, &error_statistics)) return NULL;
   calculate_pdfs(dfactor/100. + 0.0005,&radar_data, &error_statistics);

   for ( ii=0; ii<n; ii++)  {
         cout[ii]= radar_data.rain_rate_pdf[ii];
   }

   deallocate(&error_statistics, &radar_data);
   return PyArray_Return(vecout);
}

int  not_doublevector(PyArrayObject *vec)  {
   if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
      PyErr_SetString(PyExc_ValueError,
         "In not_doublevector: array must be of type Float and 1 dimensional (n).");
      return 1;  }
   return 0;
}

/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
   int i,n;
   
   n=arrayin->dimensions[0];
   return (double *) arrayin->data;  /* pointer to arrayin data as double */
}
