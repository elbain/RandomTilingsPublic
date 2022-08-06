#ifndef TRIANGLEDIMERKERNEL_CUH_
#define TRIANGLEDIMERKERNEL_CUH_

#define BLOCK_DIM 8

#include <curand_mtgp32.h>


void RotateLozenges(dim3 block_size, dim3 grid_size, curandStateMtgp32* d_status, int* tiling, const int N, const int t, const int c);

void UpdateLozengesFlipped(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int t, const int c);

void UpdateLozenges0(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2,  int* tiling3, const int N);

void UpdateLozenges1(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2,  int* tiling3, const int N);

void UpdateLozenges2(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2,  int* tiling3, const int N);

void UpdateLozenges(dim3 block_size, dim3 grid_size, int* tiling1, int* tiling2, int* tiling3, const int N, const int t);

void UpdateTriangleUFromLozenges(dim3 block_size, dim3 grid_size, int* tilingH, int* tilingL, int* tilingU, const int N);

void UpdateButterflysHFromLozenge(dim3 block_size, dim3 grid_size, int* tilingBH, int* tilingLH, int* tilingLL, int* tilingLR, const int N);

void UpdateButterflysLFromLozenge(dim3 block_size, dim3 grid_size, int* tilingBL, int* tilingLH, int* tilingLL, int* tilingLR, const int N);

void UpdateButterflysRFromLozenge(dim3 block_size, dim3 grid_size, int* tilingBR, int* tilingLH, int* tilingLL, int* tilingLR, const int N);

void RotateTriangles(dim3 block_size, dim3 grid_size, curandStateMtgp32* d_status, int* tiling, const int N, const int c);

void UpdateTrianglesFlipped0(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int c);

void UpdateTrianglesFlipped1(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int c);

void UpdateTriangles(dim3 block_size, dim3 grid_size, int* tiling1, int* tiling2, const int N, const int t);

void UpdateLozengeHFromTriangles(dim3 block_size, dim3 grid_size, int* tilingU, int* tilingD, int* tilingH, const int N);

void UpdateLozengeLFromTriangles(dim3 block_size, dim3 grid_size, int* tilingU, int* tilingD, int* tilingL, const int N);

void UpdateLozengeRFromTriangles(dim3 block_size, dim3 grid_size, int* tilingU, int* tilingD, int* tilingR, const int N);

void RotateButterflys(dim3 block_size, dim3 grid_size, curandStateMtgp32* d_status, int* tiling, const int N, const int t, const int p1, const int p2);

void UpdateButterflysFlippedH1(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1, const int p2);

void UpdateButterflysFlippedH21(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateButterflysFlippedH22(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateButterflysFlippedH23(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateButterflysFlippedL1(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1, const int p2);

void UpdateButterflysFlippedL21(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateButterflysFlippedL22(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateButterflysFlippedL23(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateButterflysFlippedR1(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1, const int p2);

void UpdateButterflysFlippedR21(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateButterflysFlippedR22(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateButterflysFlippedR23(dim3 block_size, dim3 grid_size, int* tiling, const int N, const int p1);

void UpdateLozengeFromButterflysH(dim3 block_size, dim3 grid_size, int* tilingBH, int* tilingLH, int* tilingLL, int* tilingLR, const int N);

void UpdateLozengeFromButterflysL(dim3 block_size, dim3 grid_size, int* tilingBL, int* tilingLH, int* tilingLL, int* tilingLR, const int N);

void UpdateLozengeFromButterflysR(dim3 block_size, dim3 grid_size, int* tilingBR, int* tilingLH, int* tilingLL, int* tilingLR, const int N);


#endif // TRIANGLEDIMERKERNEL_CUH_