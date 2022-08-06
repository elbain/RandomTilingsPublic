#ifndef RECTTRIANGLEKERNEL_CUH_
#define RECTTRIANGLEKERNEL_CUH_

#define BLOCK_DIM 8

#include <curand_mtgp32.h>

void flipTiles(dim3 block_size, dim3 grid_size, curandStateMtgp32* d_status, int* tiling, const int N);

void updateTiles1(dim3 block_size, dim3 grid_size, int* tiling1, int* tiling2, const int N, const int t);

void updateTiles2(dim3 block_size, dim3 grid_size, int* tiling1, int* tiling2, const int N, const int t);

#endif // RECTTRIANGLEKERNEL_CUH_
