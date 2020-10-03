#pragma once


uint* randomPermutation(uint n);
struct Colorer* CpuColor(struct Graph* graph);
void checkColors(struct Colorer* colorer, struct Graph* graph, bool verbose);

