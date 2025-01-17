#include "GraphAux.h"
#include "Vector.h"
#include <stdlib.h>
#include <stdio.h>
#include "common.h"


Graph* graphInit(uint numberNodes) {
	Graph* graph;
	//allocate sizes for the graph
	CHECK(cudaMallocManaged(&graph, sizeof(Graph), cudaMemAttachGlobal)); 
	CHECK(cudaMallocManaged(&(graph->cumDegs), (numberNodes + 1) * sizeof(uint), cudaMemAttachGlobal));

	//allocate size for the neighs

	graph->nodeSize = numberNodes;
	graph->edgeSize = 0;
	graph->maxDeg = 0;
	graph->minDeg = 0;
	graph->meanDeg = 0.0f;
	graph->connected = true;
	graph->density = 0;
	return graph;
}



void randomErdosGraph(Graph* graph, float prob) {
	if (prob < 0 || prob > 1) {
		printf("[Graph] Warning: Probability not valid (set p = 0.5)!!\n"); 
		prob = 0.5;
	}
	
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
			
			if (r < prob) {
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

void writeGraphToFile(Graph* graph, char* name) { // write order from GraphAux.h
	FILE* fp = fopen(name, "wb");
	if (fp == NULL) {
		printf("file error\n");
		return;
	}
	uint a = graph->nodeSize;
	fwrite(&graph->nodeSize, sizeof(uint), 1, fp);
	fwrite(&graph->edgeSize, sizeof(uint), 1, fp);
	for (int i = 0; i < graph->nodeSize + 1; i++) {
		if (!fwrite(&graph->cumDegs[i], sizeof(uint), 1, fp)) {
			printf("write error\n");
		}
	}
	for (int i = 0; i < graph->edgeSize; i++) {
		fwrite(&graph->neighs[i], sizeof(uint), 1, fp);
	}
	fwrite(&graph->density, sizeof(float), 1, fp);
	fwrite(&graph->maxDeg, sizeof(uint), 1, fp);
	fwrite(&graph->minDeg, sizeof(uint), 1, fp);
	fwrite(&graph->meanDeg, sizeof(float), 1, fp);
	fwrite(&graph->connected, sizeof(bool), 1, fp);

	fclose(fp);
}

Graph* readGraphFromFile(char* name) {
	FILE* fp = fopen(name, "rb");
	if (fp == NULL) {
		printf("file error\n");
		exit(2);
	}
	uint numberNodes;
	fread(&numberNodes, sizeof(uint), 1, fp);
	Graph* graph = graphInit(numberNodes);
	graph->nodeSize = numberNodes;
	fread(&graph->edgeSize, sizeof(uint), 1, fp);
	for (int i = 0; i < graph->nodeSize + 1; i++) {
		fread(&graph->cumDegs[i], sizeof(uint), 1, fp);
	}
	CHECK(cudaMallocManaged(&(graph->neighs), graph->edgeSize * sizeof(node), cudaMemAttachGlobal));
	for (int i = 0; i < graph->edgeSize; i++) {
		fread(&graph->neighs[i], sizeof(node), 1, fp);
	}
	fread(&graph->density, sizeof(float), 1, fp);
	fread(&graph->maxDeg, sizeof(uint), 1, fp);
	fread(&graph->minDeg, sizeof(uint), 1, fp);
	fread(&graph->meanDeg, sizeof(float), 1, fp);
	fread(&graph->connected, sizeof(bool), 1, fp);


	fclose(fp);
	return graph;
}
