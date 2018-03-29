/* calculate_pdfs
 *
 * Calculate the rain rate PDFs for every pixel in the radar image.
 *
 * Tim Hume.
 * 20 September 2007.
 */

#include "radar_error.h"

int calculate_pdfs(double percentile, struct radar *radar_data, struct errorstats *error_statistics){
   size_t   pix;                     /* Pixel we are currently computing the PDF for.*/
   size_t   stn;                     /* Station number.*/

   double   stn_dist;                  /* Distance from the pixel to a rain gauge.*/
   double   min_dist;                  /* Distance to the closest rain gauge.*/
   size_t   closest_stn;               /* Closest rain gauge number.*/

   double   mu;                        /* Mean of ln(Rr/Rg)*/
   double   sigma;                     /* Standard deviation of ln(Rr/Rg)*/
   double   error;                     /* Value of Rr/Rt for the given percentile.*/
   int      return_status = 0;

   for (pix=0; pix<(radar_data->nx*radar_data->ny); ++pix){
      /*
       * Determine which station (which we have error statistics for) is closest.*/
      min_dist   = FLT_MAX;            /* Every station should be closer than this. */
      closest_stn   = 0;
      for (stn=0; stn<error_statistics->nstns; ++stn){
         stn_dist   = distance(*(radar_data->lon+pix), *(radar_data->lat+pix), *(error_statistics->lon+stn),
               *(error_statistics->lat+stn));
         if (stn_dist < min_dist){
            closest_stn   = stn;
            min_dist   = stn_dist;
         }
      }
      *(radar_data->nearest_stn + pix)   = (int)closest_stn;

      /*
       * Based on the error statistics for the closest station and the radar rain rate, calculate a PDF for the true
       * rain rate. This assumes the radar error, Rr/Rt (Rr = radar rain rate, Rt = true rain rate) has a log-normal
       * distribution which is the same as Rr/Rg (Rg = gauge rain rate).
       */
      mu      = *(error_statistics->mu + closest_stn);
      sigma   = *(error_statistics->sigma + closest_stn);
       if (*(radar_data->rain_rate+pix) < 1){
          /*
           * For radar rain rates less than 1 mm/hour, assume the radar is pretty accurate (its very likely
           * recording 0 mm/hour).
           */
          error   = 1.0;
       } else {
          /*
           * Assume the radar error (Rr/Rt) is log-normal.
           */
          error   = find_percentile_error(percentile, mu, sigma);
       }
       *(radar_data->rain_rate_pdf+pix) = *(radar_data->rain_rate+pix)/error;
   }

   return return_status;
}
