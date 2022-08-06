#ifndef DOMINOKERNEL_CUH_
#define DOMINOKERNEL_CUH_

#define BLOCK_DIM 8

#include <curand_mtgp32.h>

void RotateTiles(dim3 block_size, dim3 grid_size, curandStateMtgp32* d_status, char* tiling, const int N, const int t);

void UpdateTiles(dim3 block_size, dim3 grid_size, char* tiling, char* reftiling, const int N, const int t);


#endif // DOMINOKERNEL_CUH_
