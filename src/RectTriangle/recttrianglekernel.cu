#include "recttrianglekernel.cuh"
#include <curand_kernel.h>
#include "../common/helper_cuda.h"
#include "stdio.h"

#define wt1 1.00000
#define wt2 1.00000
#define wt3 1.00000
#define wm1 1.00000
#define wm2 1.00000
#define wm3 1.00000
#define wr1 1.00000
#define wr2 1.00000
#define wr3 1.00000

__constant__ float weights[16] = { 0.0, wt1, wt2, wt3, wr1, wr2, wr3, wr1, wr2, wr3, wm1, wm2, wm3, wm1, wm2, wm3 };

__device__ float getWeightRatio(int a, int b)
{
    return ((weights[(b & 0xf00000) / 0x100000] * weights[(b & 0x0f0000) / 0x010000] * weights[(b & 0x00f000) / 0x001000] * weights[(b & 0x000f00) / 0x000100] * weights[(b & 0x0000f0) / 0x000010] * weights[(b & 0x00000f) / 0x000001]) / (weights[(a & 0xf00000) / 0x100000] * weights[(a & 0x0f0000) / 0x010000] * weights[(a & 0x00f000) / 0x001000] * weights[(a & 0x000f00) / 0x000100] * weights[(a & 0x0000f0) / 0x000010] * weights[(a & 0x00000f) / 0x000001]));
}

// Tiling is stored on vertices of the triangular lattice. Each adjacent face if given a integer value in [0,15] based on how it is covered by the tiling. These values are stored as a six digit hexidecimal int as follows:
//     ___
//   /\   /\         d4
//  /__\./__\ -> d5      d3 ->  d5*16^5 + d4*16^4 + d3*16^3 + d2*16^2 + d1*16^1 + d0*16^0
//  \  / \  /    d2      d0
//   \/___\/         d1
//

__global__ void flipTilesKernel(curandStateMtgp32* d_status, int* tiling, const int N)
{
    // Lots of 'if' statements, is there a better way?
    // Attempts to flip all tilings of a given color. Color determined by which tiling array is given as input.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // for MTGP indexing
    int id = blockIdx.x * gridDim.y + blockIdx.y;
    float rd = curand_uniform(d_status + id);

    float bs = 0.9;
    if (i < N && j < N / 3 && rd < bs) {
        rd /= bs;
        int hexType = tiling[i * (N / 3) + j];
        int hex1; int hex2; int newType = hexType;
        float sc1, sc2, sc;
        sc1 = (rd < 0.75f)? 0.5f: 0.75f; sc2 = (rd < 0.5f) ? 0.25f : sc1; sc = (rd < 0.25f) ? 0.0f : sc2;
        if (hexType == 0x122133) { /* all triangles (1) */
            // select(a,b,c), a=value  if false, b=value if true, c=condition
            hex1 = (rd < 0.75) ? 0x8ebbe8 : 0x331221; // if 0.5 < rd < 0.75 flip to 0x8ebbe8
            hex2 = (rd < 0.5) ? 0xfc99cf : hex1; // if 0.25 < rd < 0.5 flip to 0xfc99cf
            newType = (rd < 0.25) ? 0xa7dd7a : hex2; // if 0 < rd < 0.25 flip to 0xa7dd7a, else 0x331221
        }
        else if (hexType == 0x331221) { /* all triangles (3) */
            hex1 = (rd < 0.75) ? 0x8ebbe8 : 0x122133; // if 0.25 < rd < 0.75 flip to 0x8ebbe8
            hex2 = (rd < 0.5) ? 0xfc99cf : hex1;// if 0.25 < rd < 0.5 flip to 0xfc99cf
            newType = (rd < 0.25) ? 0xa7dd7a : hex2;// if 0 < rd < 0.25 flip to 0xa7dd7a, else 0x122133
        }
        else if (hexType == 0x8ebbe8) { /* single rect (b) */
            hex1 = (rd < 0.75) ? 0x122133 : 0x331221; // if 0.5 < rd < 0.75 flip to 0x122133
            hex2 = (rd < 0.5) ? 0xfc99cf : hex1; // if 0.25 < rd < 0.5 flip to 0xfc99cf
            newType = (rd < 0.25) ? 0xa7dd7a : hex2; // if 0 < rd < 0.25 flip to 0xa7dd7a, else 0x331221
        }
        else if (hexType == 0xfc99cf) { /* single rect (c) */
            hex1 = (rd < 0.75) ? 0x8ebbe8 : 0x331221;// if 0.5 < rd < 0.75 flip to 0x8ebbe8
            hex2 = (rd < 0.5) ? 0x122133 : hex1; // if 0.25 < rd < 0.5 flip to 0x122133
            newType = (rd < 0.25) ? 0xa7dd7a : hex2;  // if 0 < rd < 0.25 flip to 0xa7dd7a, else 0x331221
        }
        else if (hexType == 0xa7dd7a) { /* single rect (a) */
            hex1 = (rd < 0.75) ? 0x8ebbe8 : 0x331221;; // if 0.5 < rd < 0.75 flip to 0x8ebbe8
            hex2 = (rd < 0.5) ? 0xfc99cf : hex1; // if 0.25 < rd < 0.5 flip to 0xfc99cf
            newType = (rd < 0.25) ? 0x122133 : hex2; // if 0 < rd < 0.25 flip to 0x122133, else 0x331221
        }

        else if (hexType == 0x47d47a) { /* rect + left rect (a) */
            newType = 0xd22a33;
            sc = 1.0;
        }
        else if (hexType == 0xd22a33) { /* triangles + left rect (a) */
            newType = 0x47d47a;
            sc = 1.0;
        }

        else if (hexType == 0xa74d74) { /* rect + right rect (a) */
            newType = 0x33a22d;
            sc = 1.0;
        }
        else if (hexType == 0x33a22d) { /* triangles + right rect (a) */
            newType = 0xa74d74;
            sc = 1.0;
        }

        else if (hexType == 0x8eb558) { /* rect + left rect (b) */
            newType = 0x331eb1;
            sc = 1.0;
        }
        else if (hexType == 0x331eb1) { /* triangles + left rect (b) */
            newType = 0x8eb558;
            sc = 1.0;
        }

        else if (hexType == 0x855be8) { /* rect + right rect (b) */
            newType = 0x1be133;
            sc = 1.0;
        }
        else if (hexType == 0x1be133) { /* triangles + right rect (b) */
            newType = 0x855be8;
            sc = 1.0;
        }

        else if (hexType == 0x6699cf) { /* rect + left rect (c) */
            newType = 0xcf1221;
            sc = 1.0;
        }
        else if (hexType == 0xcf1221) { /* triangles + left rect (c) */
            newType = 0x6699cf;
            sc = 1.0;
        }

        else if (hexType == 0xfc9966) { /* rect + right rect (c) */
            newType = 0x1221fc;
            sc = 1.0;
        }
        else if (hexType == 0x1221fc) { /* triangles + right rect (c) */
            newType = 0xfc9966;
            sc = 1.0;
        }

        float weightratio = getWeightRatio(hexType, newType);
        rd = (sc == 1.0) ? rd : ((rd - sc) * 4);  //rescale rd
        tiling[i * (N / 3) + j] = (weightratio >= rd)? newType:hexType;
    }
}

__global__ void updateTiles1Kernel(int* tiling1, int* tiling2, const int N, const int t)
{
    // nned to fix indexing
    // Updates tilings, given the state of the flipped tilings (tiling1).
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N - 1 && j < (N / 3) - 1 && i>0 && j>0) { // because we padded the arrays we can ignore the boundary
        int p1 = (i % 3 == t) ? 1 : 0;
        int p2 = (i % 3 == (t + 1) % 3) ? 1 : 0;
        tiling2[i * (N / 3) + j] = (tiling1[(i - 1) * (N / 3) + j] & 0x0000f0) * 0x010000 + (tiling1[(i - 1) * (N / 3) + j] & 0x00000f) * 0x010000 + (tiling1[i * (N / 3) + j + p1] & 0xf00000) / 0x000100 + (tiling1[(i + 1) * (N / 3) + j - p2] & 0x0f0000) / 0x000100 + (tiling1[(i + 1) * (N / 3) + j - p2] & 0x00f000) / 0x000100 + (tiling1[i * (N / 3) + j + p1] & 0x000f00) / 0x000100;
    }
}

__global__ void updateTiles2Kernel(int* tiling1, int* tiling2, const int N, const int t)
{
    // need to fix indexing
    // Updates tilings, given the state of the flipped tilings (tiling1).
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N - 1 && j < (N / 3) - 1 && i>0 && j>0) { // because we padded the arrays we can ignore the boundary
        int p1 = (i % 3 == (t + 2) % 3) ? 1 : 0;
        int p2 = (i % 3 == (t + 1) % 3 )? 1 : 0;
        tiling2[i * (N / 3) + j] = (tiling1[i * (N / 3) + j - p1] & 0x00f000) * 0x000100 + (tiling1[(i - 1) * (N / 3) + j + p2] & 0x000f00) * 0x000100 + (tiling1[(i - 1) * (N / 3) + j + p2] & 0x0000f0) * 0x000100 + (tiling1[i * (N / 3) + j - p1] & 0x00000f) * 0x000100 + (tiling1[(i + 1) * (N / 3) + j] & 0xf00000) / 0x010000 + (tiling1[(i + 1) * (N / 3) + j] & 0x0f0000) / 0x010000;
    }
}


void flipTiles(dim3 block_size, dim3 grid_size, curandStateMtgp32* d_status, int* tiling, const int N)
{
    flipTilesKernel << <grid_size, block_size >> > (d_status, tiling, N);
    getLastCudaError("flipTilesKernel launch failed");
}


void updateTiles1(dim3 block_size, dim3 grid_size, int* tiling1, int* tiling2, const int N, const int t)
{
    updateTiles1Kernel << <grid_size, block_size >> > (tiling1, tiling2, N, t);
    getLastCudaError("updateTiles1Kernel launch failed");
}

void updateTiles2(dim3 block_size, dim3 grid_size, int* tiling1, int* tiling2, const int N, const int t)
{
    updateTiles2Kernel << <grid_size, block_size >> > (tiling1, tiling2, N, t);
    getLastCudaError("updateTiles2Kernel launch failed");
}
