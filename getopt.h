#ifndef GETOPT_H

#define GETOPT_H

#include <stdbool.h>;

extern int opterr;		/* if error message should be printed */
extern int optind;		/* index into parent argv vector */
extern int optopt;		/* character checked for validity */
extern int optreset;  	/* reset getopt  */
extern char* optarg;	/* argument associated with option */

struct options {
	int numberNodes ;
	float probability ;
	bool toWrite ;
	char* fileName ;
	bool toRead ;
} options;

int getopt(int nargc, char* const nargv[], const char* ostr);
void getArgs(int argc, char* argv[]);
void initializeOptions();

#endif