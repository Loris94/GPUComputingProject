#pragma once
#include <stdbool.h>;

typedef unsigned int uint;

typedef struct Colorer {
	bool		uncoloredNodes;
	bool		misNotFound;
	uint		numOfColors;
	int*		coloring;   // each element denotes a color
} Colorer;