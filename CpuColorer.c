#pragma once

#include "Colorer.h"
#include "CpuColorer.h"
#include "GraphAux.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>



Colorer* CpuColor(Graph* graph) {
    Colorer* colorer = (Colorer*)malloc(sizeof(Colorer));
    int numberNodes = graph->nodeSize;
    colorer->coloring = (int*)malloc(sizeof(int) * numberNodes);
    colorer->uncoloredNodes = true;
    colorer->numOfColors = 0;
    memset(colorer->coloring, 0, numberNodes * sizeof(int));
    uint* permutation = randomPermutation(numberNodes);

    int highest = 0;

    //color whole graph
    for (int i = 0; i < graph->nodeSize; i++) {
        int v = permutation[i]-1;
        int color = colorWithLowest(colorer, graph, v);

        if (color != 0) {
            colorer->coloring[v] = color;
            if (color > colorer->numOfColors) {
                colorer->numOfColors = color;
            }
        }
        else {
            printf("COLOR ERROR\n");
            return;
        }

    }
    
    return colorer;
 
}

Colorer* CpuLDFColor(Graph* graph) {
    uint coloredNodes = 0;
    Colorer* colorer = (Colorer*)malloc(sizeof(Colorer));
    int numberNodes = graph->nodeSize;
    uint* weigths = cpuWeigthInit(numberNodes);
    colorer->coloring = malloc(sizeof(int) * numberNodes);
    colorer->uncoloredNodes = true;
    colorer->numOfColors = 0;
    memset(colorer->coloring, 0, numberNodes * sizeof(int));
    while (coloredNodes < graph->nodeSize) {
        int max = -1;
        int index = 0;
        for (int i = 0; i < graph->nodeSize; i++) {
            if (colorer->coloring[i] == 0) {
                int deg = graph->cumDegs[i + 1] - graph->cumDegs[i];
                if (deg > max) {
                    max = deg;
                    index = i;
                }
                else if (deg == max && weigths[i]>weigths[index]) {
                    index = i;
                }
            }
        }
        int color = colorWithLowest(colorer, graph, index);
        if (color != 0) {
            colorer->coloring[index] = color;
            coloredNodes++;
            if (color > colorer->numOfColors) {
                colorer->numOfColors = color;
            }
        }
        else {
            printf("COLOR ERROR\n");
            return;
        }
    }
    return colorer;
}

int colorWithLowest(Colorer* colorer, Graph* graph, int i) {
    uint offset = graph->cumDegs[i];
    uint deg = graph->cumDegs[i + 1] - graph->cumDegs[i];
    int lowest = 0;

    for (uint k = 1; k <= deg + 1; k++) { // <= because there are at most n+1 colors, we start from 1 because 0 is for non-colored
        bool candidate = true;
        lowest = k;
        for (uint j = 0; j < deg; j++) {
            uint neighID = graph->neighs[offset + j];

            if (colorer->coloring[neighID] == k) {
                candidate = false;
                break;
            }
        }
        if (candidate) {
            break;
        }
    }
   
    return lowest;
}


uint* randomPermutation(uint n) {
    uint* r = (uint*) malloc(n * sizeof(uint));
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

uint* cpuWeigthInit(int n) {
    uint* numbers = malloc(n * sizeof(uint));
    for (int i = 0; i < n; i++) {
        numbers[i] = rand();
    }
    return numbers;
}



void checkColors(Colorer* colorer, Graph* graph, bool verbose) {
    uint n = graph->nodeSize;
    bool flag = true;
    for (int i = 0; i < n; i++) {
        uint deg = graph->cumDegs[i + 1] - graph->cumDegs[i];
        node* neighs = graph->neighs;

        if (colorer->coloring[i] == 0 || colorer->coloring[i] == -1 || colorer->coloring[i] == -2) {
            printf("error, node %d at %d\n", i, colorer->coloring[i]);
        }
        for (int j = 0; j < deg; j++) {
            if (colorer->coloring[i] == colorer->coloring[neighs[graph->cumDegs[i] + j]]) {
                printf("\nWRONG: node %d and his neighbor %d have the color %d\n", i, neighs[graph->cumDegs[i] + j], colorer->coloring[i]);
                flag = false;
            }
        }
        if (verbose) {
            printf("%d ", colorer->coloring[i]);
        }
    }
    printf("\n");
    if (flag) {
        printf("Color Good\n");
    }
    return;
}