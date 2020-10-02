#pragma once

int* randomPermutation(int n);
void CpuColor(struct Graph* graph, struct Colorer* colorer);
void checkColors(struct Colorer* colorer, struct Graph* graph, bool verbose);
