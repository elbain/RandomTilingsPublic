#include "dominokernel.cuh"
#include <curand_kernel.h>
#include "../common/helper_cuda.h"
#include "stdio.h"

#define PROB_PROPOSE 0.8

__global__ void RotateTilesKernel(curandStateMtgp32* d_status, char* tiling, const int N, const int t)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

	// for MTGP indexing
	int id = blockIdx.x * gridDim.y + blockIdx.y;
	float rd = curand_uniform(d_status + id);

	if ((i > N - 2) | (j > N / 2 - 2)) { return; }

	if (rd < PROB_PROPOSE) {
		if (tiling[i * (N / 2) + j] == 3) { tiling[i * (N / 2) + j] = 12; }
		else if (tiling[i * (N / 2) + j] == 12) { tiling[i * (N / 2) + j] = 3; }
	}

}


// t is the parity of the tiles being updated, see how this kernel is called in the RandomWalk method
__global__ void UpdateTilesKernel(char* tiling, char* reftiling, const int N, const int t)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if ((i > N - 2) | (j > N / 2 - 2)) { return; }

	tiling[i * (N / 2) + j] = (reftiling[(i - 1) * (N / 2) + j] & 2) / 2
		+ 2 * (reftiling[(i + 1) * (N / 2) + j] & 1)
		+ (reftiling[i * (N / 2) + j - (i + t + 1) % 2] & 8) / 2
		+ 2 * (reftiling[i * (N / 2) + j + (i + t) % 2] & 4);
}


void RotateTiles(dim3 block_size, dim3 grid_size, curandStateMtgp32* d_status, char* tiling, const int N, const int t)
{
	RotateTilesKernel <<<grid_size, block_size >>> (d_status, tiling, N, t);
	getLastCudaError("RotateTilesKernel launch failed");
}


void UpdateTiles(dim3 block_size, dim3 grid_size, char* tiling, char* reftiling, const int N, const int t)
{
	UpdateTilesKernel << <grid_size, block_size>> > (tiling, reftiling, N, t);
	getLastCudaError("UpdateTilesKernel launch failed");
}
