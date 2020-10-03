
#include "GpuColorer.h"
////#include <stdio.h>
//#include "wintime.c"
//

extern "C" {
	#include "common.h"
	#include "CpuColorer.h"
	#include "GraphAux.h"
}
//
#include "Colorer.h"
////
//#include <stdlib.h>
////
#include "CudaGraphAux.h"

int main(int argc, char* argv[]) {
	
	Graph* graph = graphInit(100000); //TODO cambiare con argv
	
	double start = seconds();
	randomErdosGraph(graph, 0.0001f);
	double stop = seconds();
	printf("\nRandom Erdos graph elapsed time %f sec \n", stop - start);
	
	print(graph, 0);

	//CPU COLOR
	start = seconds();
	Colorer* colorer = CpuColor(graph);
	stop = seconds();
	printf("\nCPU Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 1);
	
	//GPU COLOR LUBY
	start = seconds();
	colorer = GpuColor(graph, 0);
	stop = seconds();
	printf("\nLuby Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 1);
	
	// JP 
	
	//GPU COLOR LDF
	start = seconds();
	colorer = GpuColor(graph, 2);
	stop = seconds();
	printf("\nLDF Colorer elapsed time %f sec \n", stop - start);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	checkColors(colorer, graph, 1);
	


	// DATASET
	// FREE E CUDAFREE
	/*double start = seconds();
	uint* perm = randomPermutation(100000);
	double stop = seconds();
	printf("\nperm elapsed time %f sec \n", stop - start);*/


	cudaDeviceReset();
	return 0;
}


