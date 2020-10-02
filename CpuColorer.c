#pragma once

#include "Colorer.h";
#include "CpuColorer.h"
#include "GraphAux.h";
#include <stdio.h>
#include <stdlib.h>



void CpuColor(Graph* graph, Colorer* colorer) {
    int numberNodes = graph->nodeSize;
    colorer->coloring = (int*)malloc(sizeof(int) * numberNodes);
    colorer->uncoloredNodes = true;
    colorer->numOfColors = 0;
    memset(colorer->coloring, 0, numberNodes * sizeof(int));
    int* permutation = randomPermutation(numberNodes);

    int highest = 0;

    //color whole graph
    for (int i = 0; i < graph->nodeSize; i++) {
        int v = permutation[i]-1;
        uint offset = graph->cumDegs[v];
        uint deg = graph->cumDegs[v + 1] - graph->cumDegs[v];
        int* neighColors = (int*)malloc(sizeof(int) * deg);
        //memset(colorer->coloring, 0, numberNodes);
        for (uint j = 0; j < deg; j++) {
            uint neighID = graph->neighs[offset + j];
            neighColors[j] = colorer->coloring[neighID];
        }

        //find lowest color available
        int lowest = 0;
        
        for (uint k = 1; k <= deg+1; k++) { // <= because there are at most n+1 colors, we start from 0 because tha 0 is for non-colored
            bool candidate = true;
            lowest = k;
            for (uint j = 0; j < deg; j++) {
                if (neighColors[j] == k) {
                    candidate = false;
                    break;
                }
            }
            if (candidate) {
                break;
            }
        }

        if (lowest!=0) {
            colorer->coloring[v] = lowest;
            if (lowest > highest) {
                highest = lowest;
            }
        }
        else {
            printf("COLOR ERROR\n");
        }
        free(neighColors);
    }
    colorer->numOfColors = highest;

 
}

int* randomPermutation(int n) {
    int* r = (int*) malloc(n * sizeof(int));
    // initial range of numbers
    for (int i = 0;i < n;++i) {
        r[i] = i + 1;
    }
    // shuffle
    for (int i = n - 1; i >= 0; --i) {
        //generate a random number [0, n-1]
        int j = rand() % (i + 1);

        //swap the last element with element at random index
        int temp = r[i];
        r[i] = r[j];
        r[j] = temp;
    }
    return r;
}

void checkColors(Colorer* colorer, Graph* graph, bool verbose) {
    uint n = graph->nodeSize;
    bool flag = true;
    for (int i = 0; i < n; i++) {
        uint deg = graph->cumDegs[i + 1] - graph->cumDegs[i];
        node* neighs = graph->neighs;

        //printf("node %d has neigh: ", i);
        //for (int j = 0; j < deg; j++) {
        //	printf("%d ", neighs[str->cumDegs[i]+j]);
        //}
        //printf("\n");
        for (int j = 0; j < deg; j++) {
            if (colorer->coloring[i] == colorer->coloring[neighs[graph->cumDegs[i] + j]]) {
                printf("\nWRONG: node %d and his neighbor %d have the color %d\n", i, neighs[graph->cumDegs[i] + j], colorer->coloring[i]);
                flag = false;
            }
        }
    }
    if (flag) {
        printf("\nColor Good\n");
    }
    return;
}