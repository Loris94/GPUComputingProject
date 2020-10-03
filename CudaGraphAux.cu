#include "CudaGraphAux.h"
#include "common.h"

Graph* graphInit(uint numberNodes) {
	Graph* graph;
	//allocate sizes for the graph
	//graph->cumDegs = (uint*)malloc(sizeof(uint) * numberNodes);
	CHECK(cudaMallocManaged(&graph, sizeof(Graph))); //TODO controllare prestazioni
	CHECK(cudaMallocManaged(&(graph->cumDegs), (numberNodes + 1) * sizeof(uint)));

	//allocate size for the neighs
	//graph->neighs = (node*)malloc(sizeof(node) * numberNodes);

	graph->nodeSize = numberNodes;
	graph->edgeSize = 0;
	graph->maxDeg = 0;
	graph->minDeg = 0;
	graph->meanDeg = 0.0f;
	graph->connected = true;
	graph->density = 0;
	return graph;
}