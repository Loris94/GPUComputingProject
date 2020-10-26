#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include "GpuColorer.h"
extern "C" {
	#include "getopt.h"
	#include "common.h"
	#include "CpuColorer.h"
	#include "GraphAux.h"
}
#include "Colorer.h"


int main(int argc, char* argv[]) {

	Graph* graph;
	Colorer* colorer;
	double start, stop;

	initializeOptions();

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
	checkColors(colorer, graph, 0);
	cudaFree(colorer);

	//CPU LDF COLOR
	start = seconds();
	colorer = CpuLDFColor(graph);
	stop = seconds();
	printf("\nCPU LDF Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 0);
	cudaFree(colorer);

	//GPU COLOR LUBY
	start = seconds();
	colorer = GpuColor(graph, 0);
	stop = seconds();
	printf("\nLuby Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 0);
	cudaFree(colorer);
	
	//GPU COLOR JP
	start = seconds();
	colorer = GpuColor(graph, 1);
	stop = seconds();
	printf("\nJP Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 0);
	cudaFree(colorer);
	
	//GPU COLOR LDF
	start = seconds();
	colorer = GpuColor(graph, 2);
	stop = seconds();
	printf("\nLDF Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 0);
	cudaFree(colorer);


	// STREAMS
	// FREE E CUDAFREE


	cudaDeviceReset();
	return 0;
}








