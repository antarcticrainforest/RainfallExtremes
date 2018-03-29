/* radar_error.h
 *
 * Include file for the radar_error software.
 * 
 * Tim Hume.
 * 20 September 2007.
 */

#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <float.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

struct radar{
   size_t   nx;                     /* Number of pixels in the east-west direction.                        */
   size_t   ny;                     /* Number of pixels in the north-south direction. */
   double   *rain_rate;               /* Rain rates, in mm/hour at each radar pixel. */
   double   *lon;                  /* Longitude of each radar pixel. */
   double   *lat;                  /* Latitude of each radar pixel.  */
   int      *nearest_stn;            /* Nearest station to the pixel. */
   double   *rain_rate_pdf;         /* Holds the rain rate PDF for each pixel.  */
};

struct errorstats{
   size_t   nstns;                  /* Number of rain gauge stations where we have radar error statistics. */
   double   *lon;                  /* Longitude of each gauge.*/
   double   *lat;                  /* Latitude of each gauge. */
   double   *mu;                  /* Mean (in log space) of the log normal error distribution at each gauge. */
   double   *sigma;                  /* Standard deviation of the log normal error distribution at each gauge. */
};


/*
 * Function prototypes.
 */
 static PyObject *err(PyObject *self, PyObject *args);
int      read_error_stats(const char*, struct errorstats *);
void     deallocate(struct errorstats*, struct radar *);
int      calculate_pdfs(double, struct radar *, struct errorstats *);
double   distance(double, double, double, double);
double   find_percentile_error(double, double, double);
double   lncdf(double, double, double);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int  not_doublevector(PyArrayObject *vec);
