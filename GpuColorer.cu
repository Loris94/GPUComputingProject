
#include "GpuColorer.h"
#include "GraphAux.h"
#include <stdio.h>
#include <stdlib.h>


#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>

extern "C" {
	#include "CpuColorer.h"
	#include "common.h"
}


#define THREADxBLOCK 128


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
	uint* permutation;
	

	dim3 threads(THREADxBLOCK);
	dim3 blocks((graph->nodeSize + threads.x - 1) / threads.x, 1, 1);
	uint seed = 0;
	

	// start coloring (dyn. parall.)
	switch (type) {
	case 0:
		CHECK(cudaMallocManaged(&permutation, n * sizeof(uint)));
		managedRandomPermutation(permutation, n);
		LubyColorer <<< 1, 1 >>> (colorer, graph, permutation);
		cudaDeviceSynchronize();
		break;
	case 1:
		CHECK(cudaMalloc((void**)&states, n * sizeof(curandState_t)));
		CHECK(cudaMalloc((void**)&weigths, n * sizeof(uint)));
		init << < blocks, threads >> > (seed, states, weigths, n);
		//LubyJPcolorer << < 1, 1 >> > (colorer, graph, permutation);
		cudaFree(states);
		cudaFree(weigths);
		break;
	case 2:
		CHECK(cudaMalloc((void**)&states, n * sizeof(curandState_t)));
		CHECK(cudaMalloc((void**)&weigths, n * sizeof(uint)));
		init << < blocks, threads >> > (seed, states, weigths, n);
		cudaDeviceSynchronize();
		LDFcolorer << < 1, 1 >> > (colorer, graph, weigths);
		cudaDeviceSynchronize();
		cudaFree(states);
		cudaFree(weigths);
		break;
	}
	
	return colorer;
}

/*
* Luby MIS colorer
*/
__global__ void LubyColorer(Colorer * colorer, Graph * graph, uint * permutation) {
	dim3 threads(THREADxBLOCK);
	dim3 blocks((graph->nodeSize + threads.x - 1) / threads.x, 1, 1);

	colorer->numOfColors = 0;
	// loop on ISs covering the graph
	while (colorer->uncoloredNodes) {
		colorer->uncoloredNodes = false;
		colorer->numOfColors++;
		while (colorer->misNotFound) {
			colorer->misNotFound = false;
			LubyfindMIS <<< blocks, threads >>> (colorer, graph, permutation);
			cudaDeviceSynchronize();
			RemoveNeighs <<< blocks, threads >>> (colorer, graph, permutation);
			cudaDeviceSynchronize();
		}
		colorMIS <<< blocks, threads >>> (colorer, graph, permutation);
		cudaDeviceSynchronize();
	}
}

__global__ void LubyfindMIS(Colorer* colorer, Graph* graph, uint* permutation) {
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	uint numColors = colorer->numOfColors;

	if (idx >= graph->nodeSize || colorer->coloring[idx] != 0) {
		return;
	}
		
	uint offset = graph->cumDegs[idx];
	uint deg = graph->cumDegs[idx + 1] - graph->cumDegs[idx];

	for (uint j = 0; j < deg; j++) {
		uint neighID = graph->neighs[offset + j];

		if ((colorer->coloring[neighID] == 0 || colorer->coloring[neighID] == -1) && (permutation[idx] < permutation[neighID])) {
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

__global__ void colorMIS(Colorer* colorer, Graph* graph, uint* weights) {

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

void managedRandomPermutation(uint* permutation, uint n) {
	//uint* permutation = (uint*)malloc(n * sizeof(uint));
	// initial range of numbers
	for (int i = 0;i < n;++i) {
		permutation[i] = i + 1;
	}
	// shuffle
	for (int i = n - 1; i >= 0; --i) {
		//generate a random number [0, n-1]
		int j = rand() % (i + 1);

		//swap the last element with element at random index
		int temp = permutation[i];
		permutation[i] = permutation[j];
		permutation[j] = temp;
	}
	
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

/**
*LDF colorer
*/
__global__ void LDFcolorer(Colorer* colorer, Graph* graph, uint* weights) {
	dim3 threads(THREADxBLOCK);
	dim3 blocks((graph->nodeSize + threads.x - 1) / threads.x, 1, 1);

	// loop on ISs covering the graph
	colorer->numOfColors = 0;
	while (colorer->uncoloredNodes) {
		colorer->uncoloredNodes = false;
		colorer->numOfColors++;
		LDFfindIS <<< blocks, threads >>> (colorer, graph, weights);
		cudaDeviceSynchronize();
		colorIS <<< blocks, threads >>> (colorer, graph, weights);
		cudaDeviceSynchronize();
	}
	
}

/**
 * find an IS
 */
__global__ void LDFfindIS(Colorer* colorer, Graph* graph, uint* weights) {
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graph->nodeSize)
		return;

	if (colorer->coloring[idx])
		return;

	uint offset = graph->cumDegs[idx];
	uint deg = graph->cumDegs[idx + 1] - graph->cumDegs[idx];

	for (uint j = 0; j < deg; j++) {

		uint neighID = graph->neighs[offset + j];
		uint degNeigh = graph->cumDegs[neighID + 1] - graph->cumDegs[neighID];


		if (colorer->coloring[neighID] <= 0 && ((deg < degNeigh) || ((deg == degNeigh) && weights[idx] < weights[neighID]))) {
			colorer->uncoloredNodes = true;
			return;
		}
	}

	colorer->coloring[idx] = -1;

}

/**
 * color an IS
 */
__global__ void colorIS(Colorer* colorer, Graph* graph, uint* weights) {

	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (colorer->coloring[idx] == -1 && idx < graph->nodeSize) {
		colorer->coloring[idx] = colorer->numOfColors;
	}
	else {
		return;
	}

}



