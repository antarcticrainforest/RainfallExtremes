/* radar_error
 *
 * This program produces pixel by pixel PDFs of the radar rain rate error.
 *
 * Tim Hume.
 * 20 September 2007.
 */

#include "radar_error.h"

int main (int argc, char *argv[]){
   struct cli         cmdline;                  /* Holds the command line arguments. */
   struct errorstats   error_statistics;            /* Error statistics at rain gauge pixels. */
   struct radar      radar_data;                  /* Holds the radar rain rate data. */

   error_statistics.nstns      = 0;
   error_statistics.lon      = NULL;
   error_statistics.lat      = NULL;
   error_statistics.mu         = NULL;
   error_statistics.sigma      = NULL;

   radar_data.nx            = 0;
   radar_data.ny            = 0;
   radar_data.rain_rate      = NULL;
   radar_data.lon            = NULL;
   radar_data.lat            = NULL;
   radar_data.nearest_stn      = NULL;
   radar_data.rain_rate_pdf   = NULL;

   /*
    * Read the command line.
    */
   if (read_command(argc, argv, &cmdline)) goto finish;

   /*
    * Read the error statistics file (the error statistics are for the radar pixels where there is a colocated rain gauge.
    */
   if (read_error_stats(cmdline.errorstats_file, &error_statistics)) goto finish;

   /*
    * Read the radar data.
    */
   if (read_radar_data(cmdline.infile, &radar_data)) goto finish;

   /*
    * Calculate the radar rain rate PDFs.
    */
   if (calculate_pdfs(&radar_data, &error_statistics)) goto finish;

   /*
    * Write the rain rate PDFs to a file.
    */
   if (write_pdfs(cmdline.infile, cmdline.outfile, &radar_data)) goto finish;

   /*
    * Finish up.
    */
   finish:
   deallocate(&error_statistics, &radar_data);

   return 0;
}
