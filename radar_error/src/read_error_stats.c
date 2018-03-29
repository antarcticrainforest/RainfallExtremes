/* read_error_stats
 *
 * Read the error statistics file which contains the error statistics for the radar pixels which are colocated with 
 * rain gauges.
 *
 * The format of the white space delimited file must be exactly like this:
 * Column 1:   Name of the station, with no spaces
 * Column 2:   Longitude of the station (decimal degrees East).
 * Column 3:   Latitude of the station (decimal degrees North).
 * Column 4:   mu (mean of ln(Rr/Rg), floating point number).
 * Column 5:   sigma (standard deviation of ln(Rr/Rg), floating point number).
 * 
 * No line in the file must be more than 999 characters long (including the newline character).
 *
 * Tim Hume.
 * 20 September 2007.
 */

#include "radar_error.h"

int read_error_stats(const char *error_file, struct errorstats *error_statistics){
   FILE   *errfile;               /* Pointer to the error statistics file.                                    */
   char   line[1000];               /* Holds a line of input.                                                */
   int      return_status = 0;

   /*
    * Open the file for reading.
    */
   if (!(errfile = fopen(error_file, "r"))){
      fprintf(stderr,"E: Cannot open error statistics file, %s, for reading\n",error_file);
      return_status = 1;
      goto finish;
   }

   /*
    * Read the file until it is finished.
    */
   do {
      if (fgets(line, 1000, errfile)){
         /*
          * Need to increase the amount of memory allocated to the error_statistics structure.
          */
         error_statistics->nstns   ++;
         if (!(error_statistics->lon = realloc(error_statistics->lon, error_statistics->nstns * sizeof(double)))){
            fprintf(stderr,"E: Memory allocation problem.\n");
            goto finish;
         }
         if (!(error_statistics->lat = realloc(error_statistics->lat, error_statistics->nstns * sizeof(double)))){
            fprintf(stderr,"E: Memory allocation problem.\n");
            goto finish;
         }
         if (!(error_statistics->mu = realloc(error_statistics->mu, error_statistics->nstns * sizeof(double)))){
            fprintf(stderr,"E: Memory allocation problem.\n");
            goto finish;
         }
         if (!(error_statistics->sigma = realloc(error_statistics->sigma, error_statistics->nstns * sizeof(double)))){
            fprintf(stderr,"E: Memory allocation problem.\n");
            goto finish;
         }
         if (sscanf(line, "%*s %lf %lf %lf %lf",(error_statistics->lon + error_statistics->nstns - 1),
               (error_statistics->lat + error_statistics->nstns - 1), (error_statistics->mu + error_statistics->nstns - 1),
               (error_statistics->sigma + error_statistics->nstns - 1)) != 4){
            fprintf(stderr,"E: Unable to parse line %d in %s\n",(int)error_statistics->nstns, error_file);
            return_status = 1;
            goto finish;
         }
      } else if (ferror(errfile)){
         fprintf(stderr,"E: Problem encountered while reading %s\n",error_file);
         return_status = 1;
         goto finish;
      }
   } while (!feof(errfile));

   /*
    * Check that statistics for at leasts one station have been read from the file.
    */
   if (error_statistics->nstns < 1){
      fprintf(stderr,"E: Error statistics for at least one rain gauge must be presented.\n");
      return_status = 1;
   }

   finish:
   if (fclose(errfile)){
      fprintf(stderr,"E: Unable to close %s\n",error_file);
      return_status = 1;
   }

   return return_status;
}
