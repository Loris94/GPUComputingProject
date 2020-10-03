#include "GraphAux.h"
#include "Vector.h"
#include <stdlib.h>
#include <stdio.h>
#include "common.h"



//void graphInit(Graph* graph, uint numberNodes) {
//	//allocate sizes for the graph
//	//graph->cumDegs = (uint*)malloc(sizeof(uint) * numberNodes);
//	CHECK(cudaMallocManaged(graph, sizeof(Graph), cudaMemAttachGlobal)); //TODO controllare prestazioni
//	CHECK(cudaMallocManaged(&(graph->cumDegs), (numberNodes + 1) * sizeof(uint), cudaMemAttachGlobal));
//
//	//allocate size for the neighs
//	//graph->neighs = (node*)malloc(sizeof(node) * numberNodes);
//	
//	graph->nodeSize = numberNodes;
//	graph->edgeSize = 0;
//	graph->maxDeg = 0;
//	graph->minDeg = 0;
//	graph->meanDeg = 0.0f;
//	graph->connected = true;
//	graph->density = 0;
//	return;
//}



void randomErdosGraph(Graph* graph, float prob) {
	if (prob < 0 || prob > 1) {
		printf("[Graph] Warning: Probability not valid (set p = 0.5)!!\n"); //TODO
	}
	
	//printf("RANDMAX: %d\n", RAND_MAX);
	//rand();
	srand(time(NULL));   // Initialization, should only be called once.
	
	uint n = graph->nodeSize;

	// allocate size for an array of vectors
	vector** edges = malloc(sizeof(vector*) * n);
	for (int i = 0; i < n; i++) {
		edges[i] = vec_init();
	}

	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			float r = (float) rand() / (float)RAND_MAX;
			//printf("rand: %d\n", r);
			if (r < prob) {
				//printf("i: %d - j: %d - edges[i].size: %d - edges[j].size: %d - &edges[i]: %x &edges[j]: %x\n", i, j, edges[i]->size, edges[j]->size, &edges[i], &edges[j]);
				vec_push(edges[i], j);
				vec_push(edges[j], i);
				graph->cumDegs[i + 1]++;
				graph->cumDegs[j + 1]++;
				graph->edgeSize += 2;
			}
		}
	}
	for (int i = 0; i < n; i++)
		graph->cumDegs[i + 1] += graph->cumDegs[i];

	// max, min, mean deg
	graph->maxDeg = 0;
	graph->minDeg = n;
	for (int i = 0; i < n; i++) {
		if (deg(graph, i) > graph->maxDeg)
			graph->maxDeg = deg(graph, i);
		if (deg(graph, i) < graph->minDeg)
			graph->minDeg = deg(graph, i);
	}
	graph->density = (float)graph->edgeSize / (float)(n * (n - 1));
	graph->meanDeg = (float)graph->edgeSize / (float)n;
	if (graph->minDeg == 0)
		graph->connected = false;
	else
		graph->connected = true;

	// manage memory for edges with cuda unified memory
	/*if (gpuenabled)
		memsetgpu(n, "edges");
	else
		graph->neighs = new node[graph->edgeSize]{ };*/

	CHECK(cudaMallocManaged(&(graph->neighs), graph->edgeSize * sizeof(node), cudaMemAttachGlobal));

	for (int i = 0; i < n; i++)
		memcpy((graph->neighs + graph->cumDegs[i]), edges[i]->data, sizeof(uint) * edges[i]->size);
}


bool isValid(Graph* graph) {
	for (int i = 0; i < graph->edgeSize; i++) {
		if (graph->neighs[i] > graph->nodeSize - 1) {
			return false;
		}
	}
		
	if (graph->cumDegs[graph->nodeSize] != graph->edgeSize)  // inconsistent number of edges
		return false;
	return true;
};

/// return the degree of node i
uint deg(Graph* graph, node i) {
	return(graph->cumDegs[i + 1] - graph->cumDegs[i]);
}


/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void print(Graph* graph, bool verbose) {
	node n = graph->nodeSize;
	printf("** Graph (num node: %d , num edges: %d)\n", n, graph->edgeSize);
	printf("         (min deg: %d, max deg: %d, mean deg: %f, connected: %d)\n", graph->minDeg, graph->maxDeg, graph->meanDeg, graph->connected);

	if (verbose) {
		for (int i = 0; i < n; i++) {
			printf("   node(%d)[%d]-> ", i, graph->cumDegs[i + 1] - graph->cumDegs[i] );
			for (int j = 0; j < graph->cumDegs[i + 1] - graph->cumDegs[i]; j++) {
				printf("%d ", graph->neighs[graph->cumDegs[i] + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}
