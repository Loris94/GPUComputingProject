#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>



#include "GpuColorer.h"
////#include <stdio.h>
//#include "wintime.c"
//

extern "C" {

	#include "getopt.h"
	#include "common.h"
	#include "CpuColorer.h"
	#include "GraphAux.h"
}
//
#include "Colorer.h"
////
//#include <stdlib.h>
////

struct options{
	int numberNodes = 0;
	float probability = 0;
	bool toWrite = false;
	char* fileName = "";
	bool toRead = 0;
} options;

void getArgs(int argc, char* argv[]);


int main(int argc, char* argv[]) {

	Graph* graph;
	Colorer* colorer;
	double start, stop;

	getArgs(argc, argv);

	if (options.toRead) {
		start = seconds();
		graph = readGraphFromFile(options.fileName);
		stop = seconds();
		printf("\nGraph read graph elapsed time %f sec \n", stop - start);
	}
	else {
		graph = graphInit(options.numberNodes); 	
		start = seconds();
		randomErdosGraph(graph, options.probability);
		stop = seconds();
		printf("\nRandom Erdos graph elapsed time %f sec \n", stop - start);
	}

	if (options.toWrite) {
		start = seconds();
		writeGraphToFile(graph, options.fileName);
		stop = seconds();
		printf("\nGraph write elapsed time %f sec \n", stop - start);
	}

	print(graph, 0);

	//CPU COLOR
	start = seconds();
	colorer = CpuColor(graph);
	stop = seconds();
	printf("\nCPU Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 1);
	cudaFree(colorer);

	
	//GPU COLOR LUBY
	start = seconds();
	colorer = GpuColor(graph, 0);
	stop = seconds();
	printf("\nLuby Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 1);
	cudaFree(colorer);
	
	// JP 
	
	//GPU COLOR LDF
	start = seconds();
	colorer = GpuColor(graph, 2);
	stop = seconds();
	printf("\nLDF Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 1);
	cudaFree(colorer);


	// DATASET
	// FREE E CUDAFREE

	cudaDeviceReset();
	return 0;
}

void getArgs(int argc, char* argv[]) {

	int c;

	while ((c = getopt(argc, argv, "n:p:w:r:")) != -1)
		switch (c)
		{
		case 'n':
			options.numberNodes = atoi(optarg);
			if (options.toRead) {
				printf("Usage: you can't use -n/-p and -r together");
				exit(1);
			}
			else if (!options.numberNodes || options.numberNodes <= 0) {
				printf("Error in number nodes arg, please use only positive numbers");
			}
			break;
		case 'p':
			options.probability = atof(optarg);
			if (options.toRead) {
				printf("Usage: you can't use -n/-p and -r together");
				exit(1);
			}
			else if (!options.probability || options.probability <= 0) {
				printf("Error in number probability arg, please use only positive numbers");
			}
			break;
		case 'w':
			if (options.toRead) {
				printf("Usage: you can't use -w and -r together");
				exit(1);
			}
			options.toWrite = true;
			options.fileName = optarg;
			break;
		case 'r':
			if (options.toWrite) {
				printf("Usage: you can't use -w and -r together");
				exit(1);
			}
			else if (options.numberNodes || options.probability) {
				printf("Usage: you can't use -n/-p and -r together");
				exit(1);
			}
			options.toRead = true;
			options.fileName = optarg;
			break;
		case '?':
			/*if (optopt == 'r')
				fprintf(stderr, "Option -%c requires an argument.\n", optopt);
			else if (isprint(optopt))
				fprintf(stderr, "Unknown option `-%c'.\n", optopt);
			else
				fprintf(stderr,
					"Unknown option character `\\x%x'.\n",
					optopt);*/
			return;
		default:
			abort();
		}

	// Check mandatory parameters:
	if ((!options.numberNodes || !options.probability) && !options.toRead) {
		printf("Usage: you need to use -n/-p or -r");
		exit(1);
	}
	else if ((!options.numberNodes && options.probability) || (options.numberNodes && !options.probability)) {
		printf("Usage: -n/-p arg is missing");
		exit(1);
	}
	printf("options: %s - %d - %d - %d - %f\n", options.fileName, options.toRead, options.toWrite, options.numberNodes, options.probability);
	return;
}






