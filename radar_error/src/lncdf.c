/* lncdf
 *
 * Calculate the long-normal CDF.
 *
 * Tim Hume.
 * 21 September 2007.
 */

#include "radar_error.h"

double lncdf(double x, double mu, double sigma){
   double  cdf;

   if (x > 0){
      cdf   = 0.5 + 0.5*erf((log(x) - mu)/(sigma*sqrt(2)));
   } else {
      fprintf(stderr,"W: log-normal CDF undefined for x<=0\n");
      cdf   = -1;
   }

   return cdf;
}
