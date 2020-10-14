#pragma once
#include <stdbool.h>

typedef unsigned int node;
typedef unsigned int uint;

typedef struct Graph {
	uint nodeSize;					// num of graph nodes
	uint edgeSize;					// num of graph edges
	uint* cumDegs;					// cumsum of node degrees
	node* neighs;					// list of neighbors for all nodes (edges)
	float density;					// Probability of an edge (Erdos graph)
	uint maxDeg;
	uint minDeg;
	float meanDeg;
	bool connected;
} Graph;

Graph* graphInit(uint numberNodes);
void randomErdosGraph(Graph* graph, float prob);
void print(Graph* graph, bool verbose);
bool isValid(Graph* graph);
void writeGraphToFile(Graph* graph, char* name);
Graph* readGraphFromFile(char* name);
