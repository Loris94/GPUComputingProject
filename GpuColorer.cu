#include "GpuColorer.h"
#include "GraphAux.h"
#include <stdio.h>
#include <stdlib.h>


#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>


#define THREADxBLOCK 128

void graphInit(Graph* graph, uint numberNodes) {
	//allocate sizes for the graph
	//graph->cumDegs = (uint*)malloc(sizeof(uint) * numberNodes);
	CHECK(cudaMallocManaged(&graph, sizeof(Graph), cudaMemAttachGlobal)); //TODO controllare prestazioni
	CHECK(cudaMallocManaged(&(graph->cumDegs), (numberNodes + 1) * sizeof(uint), cudaMemAttachGlobal));

	//allocate size for the neighs
	//graph->neighs = (node*)malloc(sizeof(node) * numberNodes);

	graph->nodeSize = numberNodes;
	graph->edgeSize = 0;
	graph->maxDeg = 0;
	graph->minDeg = 0;
	graph->meanDeg = 0.0f;
	graph->connected = true;
	graph->density = 0;
	return;
}

Colorer* GpuColor(Graph* graph, int type) {
	Colorer* colorer;

	//Graph* graph = graphInit(graph, 1000);
	CHECK(cudaMallocManaged(&colorer, sizeof(Colorer)));
	uint n = graph->nodeSize;
	colorer->uncoloredNodes = true;
	colorer->misNotFound = true;
	// cudaMalloc for arrays of struct Coloring;
	CHECK(cudaMallocManaged(&(colorer->coloring), n * sizeof(uint)));
	memset(colorer->coloring, 0, n);
	
	// allocate space on the GPU for the random states
	curandState_t* states;
	uint* weigths;
	cudaMalloc((void**)&states, n * sizeof(curandState_t));
	cudaMalloc((void**)&weigths, n * sizeof(uint));
	dim3 threads(THREADxBLOCK);
	dim3 blocks((graph->nodeSize + threads.x - 1) / threads.x, 1, 1);
	uint seed = 0;
	//init << < blocks, threads >> > (seed, states, weigths, n);

	// start coloring (dyn. parall.)
	switch (type) {
	case 0:
		LubyColorer <<< 1, 1 >>> (colorer, graph, weigths);
		cudaDeviceSynchronize();
		break;
	case 1:
		//LubyJPcolorer << < 1, 1 >> > (colorer, graph, weigths);
		break;
	case 2:
		//LDFcolorer << < 1, 1 >> > (colorer, graph, weigths);
		break;
	}
	cudaFree(states);
	cudaFree(weigths);
	return colorer;
}

/**
 *  this GPU kernel takes an array of states, and an array of ints, and puts a random int into each
 */
__global__ void init(uint seed, curandState_t* states, uint* numbers, uint n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > n)
		return;
	curand_init(seed, idx, 0, &states[idx]);
	numbers[idx] = curand(&states[idx]);
}

/*
* Luby MIS colorer
*/
__global__ void LubyColorer(Colorer * colorer, Graph * graph, uint * weights) {
	dim3 threads(THREADxBLOCK);
	dim3 blocks((graph->nodeSize + threads.x - 1) / threads.x, 1, 1);

	// loop on ISs covering the graph
	while (colorer->uncoloredNodes) {
		colorer->uncoloredNodes = false;
		colorer->numOfColors++;
		while (colorer->misNotFound) {
			colorer->misNotFound = false;
			LubyfindMIS <<< blocks, threads >>> (colorer, graph, weights);
			cudaDeviceSynchronize();
			RemoveNeighs <<< blocks, threads >>> (colorer, graph, weights);
			cudaDeviceSynchronize();
		}
		Color <<< blocks, threads >>> (colorer, graph, weights);
		cudaDeviceSynchronize();
	}
}

__global__ void LubyfindMIS(Colorer* colorer, Graph* graph, uint* weights) {
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	uint numColors = colorer->numOfColors;

	if (idx >= graph->nodeSize || colorer->coloring[idx] != 0) {
		return;
	}
		
	uint offset = graph->cumDegs[idx];
	uint deg = graph->cumDegs[idx + 1] - graph->cumDegs[idx];

	for (uint j = 0; j < deg; j++) {
		uint neighID = graph->neighs[offset + j];
		uint degNeigh = weights[neighID];

		//if (colorer->coloring[neighID] == 0 && ((weights[idx] < weights[neighID]) || ((weights[idx] == weights[neighID]) && idx < neighID))) {
		if ((colorer->coloring[neighID] == 0 || colorer->coloring[neighID] == -1) && (idx < neighID)) {
			colorer->uncoloredNodes = true;
			colorer->misNotFound = true;
			return;
		} 
		
	}
	colorer->coloring[idx] = -1;
	return;
}

__global__ void RemoveNeighs(Colorer* colorer, Graph* graph, uint* weights) {
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (idx >= graph->nodeSize || colorer->coloring[idx] != 0) {
		return;
	}
	
	uint offset = graph->cumDegs[idx];
	uint deg = graph->cumDegs[idx + 1] - graph->cumDegs[idx];
		
	for (uint j = 0; j < deg; j++) {
		uint neighID = graph->neighs[offset + j];

		if (colorer->coloring[neighID] == -1) {
			colorer->coloring[idx] = -2;
			return;
		}
	}

}

__global__ void Color(Colorer* colorer, Graph* graph, uint* weights) {

	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	colorer->misNotFound = true;
	if (colorer->coloring[idx] == -1 && idx < graph->nodeSize) {
		colorer->coloring[idx] = colorer->numOfColors;
	}
	else if (colorer->coloring[idx] == -2 && idx < graph->nodeSize){
		colorer->coloring[idx] = 0;
	}
	else {
		return;
	}

}