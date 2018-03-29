/* distance
 *
 * Calculate the distance between two points (using whatever metric the user codes up in here).
 *
 * Tim Hume.
 * 20 September 2007.
 */

#include "radar_error.h"

double distance(double lon1, double lat1, double lon2, double lat2){
   double   distance;            /* Distance between points (lon1, lat1) and (lon2, lat2) */
   double   r2d = 57.2958;         /* Factor to convert from radians to degrees   */

   distance = 1.852*60*r2d * acos(sin(lat1/r2d)*sin(lat2/r2d) + cos(lat1/r2d)*cos(lat2/r2d)*cos((lon2-lon1)/r2d));

   return distance;
}
