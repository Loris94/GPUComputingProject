#pragma once

#include "Colorer.h"
#include "common.h"
#include <curand_kernel.h>



struct Colorer* GpuColor(struct Graph* graph, int type);
__global__ void LubyColorer(struct Colorer* col, struct  Graph* graph, uint* weights);
__global__ void init(uint seed, curandState_t* states, uint* numbers, uint n);
__global__ void LubyfindMIS(struct Colorer* colorer, struct Graph* str, uint* weights);
__global__ void Color(struct Colorer* colorer, struct Graph* graph, uint* weights);
__global__ void RemoveNeighs(struct Colorer* colorer, struct Graph* graph, uint* weights);