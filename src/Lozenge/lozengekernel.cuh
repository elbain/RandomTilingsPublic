#ifndef LOZENGEKERNEL_CUH_
#define LOZENGEKERNEL_CUH_

#define BLOCK_DIM 8

#include <curand_mtgp32.h>

void RotateTiles(dim3 block_size, dim3 grid_size, curandStateMtgp32* d_status, char* tiling, const int N);

void UpdateTiles(dim3 block_size, dim3 grid_size, char* tiling1, char* tiling2, char* tiling3, const int N, const int t);


#endif // LOZENGEKERNEL_CUH_