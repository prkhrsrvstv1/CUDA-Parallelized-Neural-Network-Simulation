# CUDA-Parallelized-Neural-Network-Simulation

## Sequential_CPU
Contains the original starting simulation code.
#### `LIFnetwork.c`
C code for running neuron-network simulations sequentially
#### `Create_Connected_Network.c`
Helper function for `LIFnetwork.c` to create a random neuron network

## Parallel_GPU
#### `LIFnetwork_Parallel.cu`
Parallelized rewrite of `LIFnetwork.c` in CUDA

## Notes:
- Add `-lcurand` flag while compiling the CUDA file
- Presently, the block grid an blocks themselves are both 1D, they can trivially be extended to 2D or 3D for more number of simultaneous threads.
