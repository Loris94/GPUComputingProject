
#include "GpuColorer.h"


#include "wintime.h"
#include <stdio.h>


extern "C" {
	#include "CpuColorer.h"
	#include "GraphAux.h"
	#include "Colorer.h"
}
#include <stdlib.h>

#include "CudaGraphAux.h"



int main(int argc, char* argv[]) {
	Graph* graph = graphInit(1000);
	Colorer* colorer = (Colorer*)malloc(sizeof(Colorer));
	//TODO cambiare con argv
	randomErdosGraph(graph, 0.01f);
	printf("graph done\n");
	//print(graph, 1);

	//CPU COLOR
	//CpuColor(graph, colorer);
	//checkColors(colorer, graph, 1);
	//printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);


	//GPU COLOR LUBY
	colorer = GpuColor(graph, 0);
	checkColors(colorer, graph, 1);
	printf("num of colors: %d\nhighest degree: %d\n", colorer->numOfColors, graph->maxDeg);
	// LUBY
	// JP 
	// LDF
	// DATASET

	cudaDeviceReset();
	return 0;
}


