/* deallocate
 *
 * Free memory which has been allocated within the radar_error program.
 *
 * Tim Hume.
 * 20 September 2007.
 */

#include "radar_error.h"

void deallocate(struct errorstats *error_statistics, struct radar *radar_data){
   int      ii;

   if (error_statistics->lon)      free(error_statistics->lon);
   if (error_statistics->lat)      free(error_statistics->lat);
   if (error_statistics->mu)      free(error_statistics->mu);
   if (error_statistics->sigma)   free(error_statistics->sigma);

   if (radar_data->rain_rate)      free(radar_data->rain_rate);
   if (radar_data->lon)         free(radar_data->lon);
   if (radar_data->lat)         free(radar_data->lat);
   if (radar_data->nearest_stn)   free(radar_data->nearest_stn);
   if (radar_data->rain_rate_pdf) free(radar_data->rain_rate_pdf);

   error_statistics->lon      = NULL;
   error_statistics->lat      = NULL;
   error_statistics->mu       = NULL;
   error_statistics->sigma    = NULL;

   radar_data->rain_rate      = NULL;
   radar_data->lon            = NULL;
   radar_data->lat            = NULL;
   radar_data->nearest_stn    = NULL;
   radar_data->rain_rate_pdf  = NULL;

   return;
}
