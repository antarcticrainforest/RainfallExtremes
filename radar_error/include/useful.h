//This is the header guard for the useful object
//
//Useful should supply different methods that are ease the workflow 
//by supplying methods that are frequently used
//
#ifndef USEFUL_H //The header guard to prevent useful to be loaded twice
#define USEFUL_H
#include <vector>
#include <sstream>
// Define some datatypes:
// Config type
struct config {const char* precipdir;const char* varname;const char* lsm;};
//The function prototypes from useful
//
config readconf(const char* configFile, const char* precipdir, const char* varname, const char* lsm); // read configuration from a file
static char * copyString(const char * str);
float convfloat(char* arg); //convert a char from the stdin to a float
int convint(char* arg); //convert a char from the stdin to an integer
bool approxEqual(double a, double b); //check equality of two variables a and b
void GetFilenames(std::vector<std::string> &FileNames, const char* dirName);
//
#endif
