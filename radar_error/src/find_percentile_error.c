/* find_percentile_error
 *
 * This function finds the specified percentile for a log-normal distribution (which in this case, is modelling the rain rate
 * error, hence the name of the function). A bi-linear method is used to find the solution.
 *
 * Tim Hume.
 * 21 September 2007.
 */

#include "radar_error.h"

double find_percentile_error(double percentile, double mu, double sigma){
   double   error_min   = 0.01;               /* Assume the minimum error, Rr/Rt, will be greater than this. */
   double   error_max   = 100.;               /* Assume the maximum error, Rr/Rt, will be less than this. */
   double   error_mid;                     /* (error_min + error_max)/2 */
   double   error;
   double   prob_min;                     /* The probability that the error is less than error_min.*/
   double   prob_max;                     /* The probability that the error is less than error_max. */
   double   prob_mid;                     /* The probability that the error is less than error_mid. */
   double   closeness   = DBL_MAX;            /* |percentile - prob_mid| */

   prob_min   = lncdf(error_min, mu, sigma);
   prob_max   = lncdf(error_max, mu, sigma);

   /*
    * Check that the percentile does not lie outside the initial values of prob_min and prob_max.
    */
   if (percentile < prob_min){
      fprintf(stderr,"W: error for %f percentile is less than %f\n",percentile*100,error_min);
      error      = 0.01;
      closeness   = 0;
   }

   if (percentile > prob_max){
      fprintf(stderr,"W: error for %f percentile is greater than %f\n",percentile*100,error_max);
      error      = 100.;
      closeness   = 0;
   }

   /*
    * Iterate until we get the approximate error.
    */
   while (closeness > 0.001){
      error_mid   = (error_min + error_max)/2;
      prob_mid   = lncdf(error_mid, mu, sigma);
      if (prob_mid >= percentile){
         if ((prob_mid - percentile) >= (percentile - prob_min)){
            closeness   = percentile - prob_min;
            error      = error_min;
         } else {
            closeness   = prob_mid - percentile;
            error      = error_mid;
         }
         prob_max      = prob_mid;
         error_max      = error_mid;
      } else {
         if ((prob_max - percentile) >= (percentile - prob_mid)){
            closeness   = percentile - prob_mid;
            error      = error_mid;
         } else {
            closeness   = prob_max - percentile;
            error      = error_max;
         }
         prob_min      = prob_mid;
         error_min      = error_mid;
      }
   }

   return error;
}
