# Parallel Graph Coloring with CUDA

This project, based on “A Comparison of Parallel Graph Coloring Algorithms” paper, aims to implement the "Largest Degree First" algorithm with CUDA both in the sequential and parallel way and to check the speedup. Furthermore, a sequential greedy algorithm and others parallel algorithms have been implemented in order to have some alternatives to check the results with.

### Execution

After compiling the project it's possible to use the program to generate random Erdos graphs by giving in input the number of nodes n and a probability p, which corresponds to the probability an edge will be created between two nodes. An example:

```./GPUComputingProject -n 1000 -p 0.01```

It's possible to write on a file the graph that will be created by adding the flag -w [filename], the filename must contain the path:

```./GPUComputingProject -n 1000 -p 0.01 -w [path/filename]```

Or it's possible to read the graph directly from a file, if it has been created previously:

```./GPUComputingProject -r [path/filename]```

For obvious reasons it isn't possible to use -w and -r together nor -r and -n/-p together.

There are already some graphs ready for use in the graphs folder:

```./GPUComputingProject -r graphs/n300kp000001```