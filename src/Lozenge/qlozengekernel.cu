#include "lozengekernel.cuh"
#include <curand_kernel.h>
#include "../common/helper_cuda.h"
#include "stdio.h"
  
#define q 0.99  
  
__global__ void RotateTilesKernel( curandStateMtgp32 * d_status,  int* tiling, const int N) // fix all things with N  
{  
    // Attempts to rotate all tilings of a given color. Color determined by which tiling array is given as input.  
    // Tilings are stored on vertices of hexagonal lattice. There is an indicator on each adjacent edge which is nonzero if a lozenge (or equivalently a dimer) crosses the edge. The value of the indicator on each edge is:  
    //        1   2  
    //     4 __\./__ 8  
    //         / \  
    //       16   32  
      
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    int id = blockIdx.x * gridDim.y + blockIdx.y; // for MTGP indexing  
    float rd = curand_uniform(d_status + id);  
      
    if ( i < N && j < N/3) {  
        if (rd < .5) {  
            rd /= .5;  
            if (tiling[i*(N/3)+j] == 25 && q >= rd) tiling[i*(N/3)+j] = 38; // no cube to cube  
        } else {  
            rd = (rd-.5)*.5;  
            if (tiling[i*(N/3)+j] == 38 && 1/q >= rd) tiling[i*(N/3)+j] = 25; // cube to no cube  
        }  
    }  
      
}  
  
__global__ void UpdateTilesKernel( int* tiling1,  int* tiling2, int* tiling3, const int N, const int t)  
{  
    // Updates tilings, given the state of the flipped tilings (tiling1). Updates UR, L, and DR.  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if (i < N-1 && j < (N/3)-1 && i > 0 && j > 0) { // because we padded the arrays we can ignore the boundary  
        int p2 = (i%3 == (2*t+1)%3)?(1):(0);  
        int p4 = (i%3 == (2*t+2)%3)?(1):(0);  
        int p8 = (i%3 == (2*t)%3)?(1):(0);  
        int p16 = (i%3 == (2*t+1)%3)?(1):(0);  
          
        tiling2[i*(N/3)+j] &= ~((tiling2[i*(N/3)+j] & 2) + (tiling2[i*(N/3)+j] & 4) + (tiling2[i*(N/3)+j] & 32)); //zero out appropriate bits  
        tiling2[i*(N/3)+j] += (tiling1[(i-1)*(N/3)+j+p2] & 16)/8 + (tiling1[i*(N/3)+j-p4] & 8)/2 + (tiling1[(i+1)*(N/3)+j] & 1)*32; //update  
        tiling3[i*(N/3)+j] &= ~((tiling3[i*(N/3)+j] & 1) + (tiling3[i*(N/3)+j] & 8) + (tiling3[i*(N/3)+j] & 16)); //zero out appropriate bits  
        tiling3[i*(N/3)+j] += (tiling1[(i-1)*(N/3)+j] & 32)/32 + (tiling1[i*(N/3)+j+p8] & 4)*2 + (tiling1[(i+1)*(N/3)+j-p16] & 2)*8; //update  
    }  
}  
  
void RotateTiles(dim3 block_size, dim3 grid_size,  curandStateMtgp32 * d_status,  int* tiling, const int N) 
{ 
RotateTilesKernel << <grid_size, block_size >> > (d_status, tiling, N);
getLastCudaError("RotateTilesKernel launch failed");
} 
 
void UpdateTiles(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2, int* tiling3, const int N, const int t) 
{ 
UpdateTilesKernel << <grid_size, block_size >> > (tiling1, tiling2, tiling3, N, t);
getLastCudaError("UpdateTilesKernel launch failed");
} 
 

