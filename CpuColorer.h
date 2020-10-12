#pragma once


uint* randomPermutation(uint n);
uint* cpuWeigthInit(int n);
struct Colorer* CpuColor(struct Graph* graph);
struct Colorer* CpuLDFColor(struct Graph* graph);
void checkColors(struct Colorer* colorer, struct Graph* graph, bool verbose);

