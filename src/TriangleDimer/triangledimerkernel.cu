#include "triangledimerkernel.cuh"
#include <curand_kernel.h>
#include "../common/helper_cuda.h"
#include "stdio.h"
  
  
__global__ void RotateLozengesKernel( curandStateMtgp32 * d_status,  int* tiling, const int N, const int t, const int c)  
{  
    // Attempts a Lozenge type flip  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    int id = blockIdx.x * gridDim.y + blockIdx.y; // for MTGP indexing  
    float rd = curand_uniform(d_status + id);  
      
     if ( rd < .5 && i < N && j < N && ((1-(t&1))*i + ((1+t)/2) * j)%2 == c) {  
         if (tiling[i*N+j] == 5) {  
             tiling[i*N+j] = 10;  
         } else if (tiling[i*N+j] == 10) {  
             tiling[i*N+j] = 5;  
         }  
     }  
}  
  
__global__ void UpdateLozengesFlippedKernel( int* tiling, const int N, const int t, const int c)  
{  
    // Updates the state of all elements of the tiling the same as those just flipped. t is the orientation of those just flipped (0=H, 1=L, 2=R), c indicates which subset of a given orientation tries to flip.  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if (i < N-1 && j < N-1 && i>0 && j>0 && ((1-(t&1))*i + ((1+t)/2) * j)%2 != c) {  
        tiling[i*N+j] = (tiling[(i-1)*N+j+(t&1)] & 4)/4 + (tiling[(i-(2-t)/2)*N+j+1] & 8)/4 + (tiling[(i+1)*N+j-(t&1)] & 1)*4 + (tiling[(i+(2-t)/2)*N+j-1] & 2)*4;  
    }  
}  
  
__global__ void UpdateLozenges0Kernel( int* tiling1,  int* tiling2,  int* tiling3, const int N)  
{  
    // After flipping and updating tilings of type 0 (Horizontal), we now update tilings of type 1 (Left) and 2 (Right).  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if (i < N-1 && j < N-1 && i>0 && j>0 ) {  
        tiling2[i*N+j] &= ~10;  
        tiling2[i*N+j] |= (tiling1[(i+1)*N+j] & 1)*2 + (tiling1[(i+1)*N+j-1] & 1)*8;  
        tiling3[i*N+j] &= ~10;  
        tiling3[i*N+j] |= (tiling1[(i+1)*N+j] & 2) + (tiling1[i*N+j] &  8);  
    }  
}  
  
__global__ void UpdateLozenges1Kernel( int* tiling1,  int* tiling2,  int* tiling3, const int N)  
{  
    // After flipping and updating tilings of type 1 (Left), we now update tilings of type 2 (Right) and 0 (Horizontal).  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if (i < N-1 && j < N-1 && i>0 && j>0 ) {  
        tiling2[i*N+j] &= ~5;  
        tiling2[i*N+j] |= (tiling1[i*N+j] & 1) + (tiling1[(i+1)*N+j] & 1)*4;  
        tiling3[i*N+j] &= ~5;  
        tiling3[i*N+j] |= (tiling1[(i-1)*N+j] & 2)/2 + (tiling1[i*N+j] & 2)*2;  
    }  
}  
  
__global__ void UpdateLozenges2Kernel( int* tiling1,  int* tiling2,  int* tiling3, const int N)  
{  
    // After flipping and updating tilings of type 2 (Right), we now update tilings of type 0 (Horizontal) and 1 (Left).  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if (i < N-1 && j < N-1 && i>0 && j>0 ) {  
        tiling2[i*N+j] &= ~10;  
        tiling2[i*N+j] |= (tiling1[(i-1)*N+j] & 2) + (tiling1[i*N+j-1] & 2)*4;  
        tiling3[i*N+j] &= ~5;  
        tiling3[i*N+j] |= (tiling1[i*N+j] & 1) + (tiling1[i*N+j-1] & 4);  
    }  
}  
  
__global__ void UpdateLozengesKernel( int* tiling1,  int* tiling2,  int* tiling3, const int N, const int t)  
{  
    // Not used.  
    // Updates tilings, given the state of the adjacent tilings  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if (i < N-1 && j < N-1 && i>0 && j>0 ) {  
        tiling2[i*N+j] &= ~(2-(t&1) + 8-4*(t&1));  
        tiling2[i*N+j] |= (tiling1[(i+1-t)*N+j] & 1+t/2)*1 + (tiling1[(i+1-t/2)*N+j-1+(t&1)] & 1+t/2)*(4+4*((2-t)/2));  
        tiling3[i*N+j] &= ~(1+((2-t)/2) + 4+4*((2-t)/2));  
        tiling3[i*N+j] |= (tiling1[(i-(t&1)+(2-t)/2)*N+j] & 2-t/2)/(1+(t&1)) + (tiling1[i*N+j-t/2] & 4*(t/2) + 2*(t&1) + 8*((2-t)/2))*(1+(t&1));  
    }  
}  
  
__global__ void UpdateTriangleUFromLozengesKernel( int* tilingH,  int* tilingL,  int* tilingU, const int N)  
{  
    // Update Up triangles from Lozenges  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        int L1 = ((tilingL[i*N+j] & 1) == 1)?(1):(0);  
        int L2 = ((tilingH[i*N+j] & 8) == 8)?(2):(0);  
        int L3 = ((tilingH[(i+1)*N+j] & 1) == 1)?(3):(0);  
        int R1 = ((tilingL[i*N+j+1] & 1) == 1)?(1):(0);  
        int R2 = ((tilingH[(i+1)*N+j] & 2) == 2)?(2):(0);  
        int R3 = ((tilingL[i*N+j+1] & 2) == 2)?(3):(0);  
        int V1 = ((tilingL[i*N+j+1] & 4) == 4)?(1):(0);  
        int V2 = ((tilingH[(i+1)*N+j] & 8) == 8)?(2):(0);  
        int V3 = ((tilingH[(i+1)*N+j] & 4) == 4)?(3):(0);  
          
        tilingU[i*N+j] = L1+L2+L3 + 4*(R1+R2+R3) + 16*(V1+V2+V3);  
    }  
}  
  
__global__ void UpdateButterflysHFromLozengeKernel( int* tilingBH,  int* tilingLH,  int* tilingLL,  int* tilingLR, const int N)  
{  
    // Update horizontal butterflys from lozenges  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {  
        int LH = tilingLH[i*N+j];  
        int LR1 = tilingLR[(i-1)*N+j];  
        int LL1 = tilingLL[(i-1)*N+j+1];  
        int LR2 = tilingLR[i*N+j];  
        int LL2 = tilingLL[i*N+j];  
          
        int c = (LL1 & 4)/4;  
        int t1 = (LR1 & 8)/8 + 2*(LR1 & 1) + 3*(LH & 1);  
        int t4 = (LL1 & 1) + (LL1 & 2) + ((LH & 2)/2)*3;  
        int t16 = (LR2 & 2)/2 + (LR2 & 4)/2 + ((LH & 4)/4)*3;  
        int t64 = (LL2 & 4)/4 + (LL2 & 8)/4 + ((LH & 8)/8)*3;  
          
        tilingBH[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
    }  
}  
  
__global__ void UpdateButterflysLFromLozengeKernel( int* tilingBL,  int* tilingLH,  int* tilingLL,  int* tilingLR, const int N)  
{  
    // Update left butterflys from lozenges  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {  
        int LL = tilingLL[i*N+j];  
        int LH1 = tilingLH[i*N+j];  
        int LR1 = tilingLR[i*N+j];  
        int LH2 = tilingLH[(i+1)*N+j-1];  
        int LR2 = tilingLR[i*N+j-1];  
          
        int c = (LR1 & 8)/8;  
        int t1 = (LH1 & 1) + (LH1 & 2) + 3*(LL & 1);  
        int t4 = (LR1 & 2)/2 + (LR1 & 4)/2 + ((LL & 2)/2)*3;  
        int t16 = (LH2 & 4)/4 + (LH2 & 8)/4 + ((LL & 4)/4)*3;  
        int t64 = (LR2 & 8)/8 + 2*(LR2 & 1) + ((LL & 8)/8)*3;  
          
        tilingBL[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
    }  
}  
  
__global__ void UpdateButterflysRFromLozengeKernel( int* tilingBR,  int* tilingLH,  int* tilingLL,  int* tilingLR, const int N)  
{  
    // Update right butterflys from lozenges  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {  
        int LR = tilingLR[i*N+j];  
        int LH1 = tilingLH[i*N+j];  
        int LL1 = tilingLL[i*N+j+1];  
        int LH2 = tilingLH[(i+1)*N+j];  
        int LL2 = tilingLL[i*N+j];  
          
        int c = (LL1 & 8)/8;  
        int t1 = (LH1 & 1) + (LH1 & 2) + 3*(LR & 1);  
        int t4 = (LL1 & 1) + (LL1 & 2) + ((LR & 2)/2)*3;  
        int t16 = (LH2 & 4)/4 + (LH2 & 8)/4 + ((LR & 4)/4)*3;  
        int t64 = (LL2 & 4)/4 + (LL2 & 8)/4 + ((LR & 8)/8)*3;  
          
        tilingBR[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
    }  
      
}  
  
__global__ void RotateTrianglesKernel( curandStateMtgp32 * d_status,  int* tiling, const int N, const int c)  
{  
    // Attempts a Triangle type flip.  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    int id = blockIdx.x * gridDim.y + blockIdx.y; // for MTGP indexing  
    float rd = curand_uniform(d_status + id);  
      
    if ( rd < 0.5 && i < N && j < N && ((j-i)%3+3)%3 == c) {  
        if (tiling[i*N+j] == 45) {  
            tiling[i*N+j] = 54;  
        } else if (tiling[i*N+j] == 54) {  
            tiling[i*N+j] = 45;  
        }  
    }  
      
      
}  
  
__global__ void UpdateTrianglesFlipped0Kernel( int* tiling, const int N, const int c)  
{  
    // Update up triangles after up flips  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 && ((j-i)%3+3)%3 != c) {  
        bool b = (((j-i)%3+3)%3 == (c+1)%3);  
        int p = ( b)?( 1):(0);  
        int t1 = (4-3*p)*12;  
        int m1 = (4-3*p)*4;  
        int t4 = (15*p+1)*3;  
        int m4 = (15*p+1);  
        int t16 = (4-3*p)*3;  
        int m16 = (4-3*p);  
          
        tiling[i*N+j] = (tiling[(i-(1-p))*N+j-p] & t1)/m1 + ((tiling[(i-p)*N+j+1] & t4)/m4)*4 + ((tiling[(i+1)*N+j-1+p] & t16)/m16)*16;  
    }  
      
}  
  
__global__ void UpdateTrianglesFlipped1Kernel( int* tiling, const int N, const int c)  
{  
    // Update down triangles after down flips  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 && ((j-i)%3+3)%3 != c) {  
        bool b = (((j-i)%3+3)%3 == (c+1)%3);  
        int p = ( b)?( 1):(0);  
        int t1 = (4-3*p)*12;  
        int m1 = (4-3*p)*4;  
        int t4 = (15*p+1)*3;  
        int m4 = (15*p+1);  
        int t16 = (4-3*p)*3;  
        int m16 = (4-3*p);  
          
        tiling[i*N+j] = (tiling[(i+1-p)*N+j-1] & t1)/m1 + ((tiling[(i+p)*N+j+1-p] & t4)/m4)*4 + ((tiling[(i-1)*N+j+p] & t16)/m16)*16;  
    }  
      
}  
  
__global__ void UpdateTrianglesKernel( int* tiling1,  int* tiling2, const int N, const int t)  
{  
    // Update down triangles form up triangle (t=0), or up triangles from down triangles (t=1)  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        int p = (t==0)?(-1):(1);  
        int L1 = ((tiling1[i*N+j-1+t] & 48) == 16)?(1):(0);  
        int L2 = ((tiling1[i*N+j-1+t] & 3) == 3)?(2):(0);  
        int L3 = ((tiling1[i*N+j-1+t] & 12) == 8)?(3):(0);  
        int R1 = ((tiling1[i*N+j+t] & 48) == 16)?(1):(0);  
        int R2 = ((tiling1[i*N+j+t] & 3) == 3)?(2):(0);  
        int R3 = ((tiling1[i*N+j+t] & 12) == 8)?(3):(0);  
        int V1 = ((tiling1[(i+p)*N+j] & 48) == 16)?(1):(0);  
        int V2 = ((tiling1[(i+p)*N+j] & 3) == 3)?(2):(0);  
        int V3 = ((tiling1[(i+p)*N+j] & 12) == 8)?(3):(0);  
          
        tiling2[i*N+j] = L1+L2+L3 + 4*(R1+R2+R3) + 16*(V1+V2+V3);  
    }  
}  
  
__global__ void UpdateLozengeHFromTrianglesKernel( int* tilingU,  int* tilingD,  int* tilingH, const int N)  
{  
    // update horizontal lozenges from up triangles  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        int tD = tilingD[i*N+j];  
        int tU = tilingU[(i-1)*N+j];  
        int H1 = ( (tD & 48) == 32)?(1):(0);  
        int H2 = ( (tD & 48) == 48)?(1):(0);  
        int H4 = ( (tU & 48) == 48)?(1):(0);  
        int H8 = ( (tU & 48) == 32)?(1):(0);  
        tilingH[i*N+j] = H1+2*H2+4*H4+8*H8;  
    }  
}  
  
__global__ void UpdateLozengeLFromTrianglesKernel( int* tilingU,  int* tilingD,  int* tilingL, const int N)  
{  
    // update left lozenges from up triangles  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        int tD = tilingD[i*N+j];  
        int tU = tilingU[i*N+j-1];  
        int L1 = ( (tU & 12) == 4)?(1):(0);  
        int L2 = ( (tU & 12) == 12)?(1):(0);  
        int L4 = ( (tD & 3) == 1)?(1):(0);  
        int L8 = ( (tD & 3) == 2)?(1):(0);  
        tilingL[i*N+j] = L1+2*L2+4*L4+8*L8;  
    }  
}  
  
__global__ void UpdateLozengeRFromTrianglesKernel( int* tilingU,  int* tilingD,  int* tilingR, const int N)  
{  
    // update right lozenges from up triangles  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        int tD = tilingD[i*N+j];  
        int tU = tilingU[i*N+j];  
        int R1 = ( (tU & 3) == 1)?(1):(0);  
        int R2 = ( (tD & 12) == 12)?(1):(0);  
        int R4 = ( (tD & 12) == 4)?(1):(0);  
        int R8 = ( (tU & 3) == 2)?(1):(0);  
        tilingR[i*N+j] = R1+2*R2+4*R4+8*R8;  
    }  
}  
  
__global__ void RotateButterflysKernel( curandStateMtgp32 * d_status,  int* tiling, const int N, const int t, const int p1, const int p2)  
{  
    // Attempts a Butterfly type flip  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    int id = blockIdx.x * gridDim.y + blockIdx.y; // for MTGP indexing  
    float rd = curand_uniform(d_status + id);  
      
    int b1 = (1-(t&1));  
    int b2 = ((1+t)/2);  
    int b3 = (t&1);  
      
    if (rd < 0.5 && i < N && j < N  && ((b1*i + b2*j)%3) == p1 ) {  
        if (tiling[i*N+j] == 170) {  
            tiling[i*N+j] = (((b1*j + b3*i)&1) == p2)?(85):(170);  
        } else if (tiling[i*N+j] == 85) {  
            tiling[i*N+j] = (((b1*j + b3*i)&1) == p2)?(170):(85);  
        }  
    }  
}  
  
// Series of kernels for updating butterflys after butterfly flips. See triangledimer.cpp for a description.  
__global__ void UpdateButterflysFlippedH1Kernel( int* tiling, const int N, const int p1, const int p2)  
{  
    // Update same slice  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if((i%3) == p1 && (j&1) != p2) {  
            int Bf1 = tiling[i*N+j-1];  
            int Bf2 = tiling[i*N+j+1];  
              
            int c = (tiling[i*N+j]&256)/256;  
            int t1 = ((Bf1 & 12)/4 == 0)?(0):(((Bf1 & 12)/4)%3  + 1);  
            int t4 = ((Bf2 & 3) == 0)?(0):(((Bf2 & 3)+1)%3  + 1);  
            int t16 = ((Bf2 & 192)/64 == 0)?(0):(((Bf2 & 192)/64)%3 + 1);  
            int t64 = ((Bf1 & 48)/16 == 0)?(0):(((Bf1 & 48)/16 + 1)%3 + 1);  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
      
}  
  
__global__ void UpdateButterflysFlippedH21Kernel( int* tiling, const int N, const int p1)  
{  
    // Update adjacent slice partially  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if((i%3) == (p1+1)%3 ) { //up  
            int Bf = tiling[i*N+j];  
            int Bf1 = tiling[(i-1)*N+j];  
            int Bf2 = tiling[(i-1)*N+j+1];  
              
            int c = ((Bf1 & 48)/16 == 2)?(1):(0);  
            int t1 = ((Bf & 3) == 2)?((Bf & 3)):((Bf1 & 192)/192 + ((Bf1 & 48)/48)*3);  
            int t4 = ((Bf & 12)/4 == 1)?((Bf&12)/4):(((Bf2 & 192)/192)*3 + ((Bf2 & 48)/48)*2);  
            int t16 = (Bf & 48)/16;  
            int t64 = (Bf & 192)/64;  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
}  
  
__global__ void UpdateButterflysFlippedH22Kernel( int* tiling, const int N, const int p1)  
{  
    // Update adjacent slice  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if((i%3) == (p1+2)%3 ) {  
            int Bf = tiling[i*N+j];  
            int s1 = ((Bf & 3) != 2)?(1):(0);  
            int s4 = ((Bf & 12)/4 != 1)?(1):(0);  
            int Bf1 = tiling[(i-1)*N+j];  
            int Bf2 = tiling[(i-1)*N+j+1];  
            int Bf3 = tiling[(i+1)*N+j];  
            int Bf4 = tiling[(i+1)*N+j-1];  
              
            int c = ((Bf3 & 3) == 2)?(1):(0);  
            int t1 = ((Bf1 & 256) == 256)?(2):(s1*(Bf & 3));  
            int t4 = ((Bf2 & 256) == 256)?(1):(s4*((Bf & 12)/4));  
            int t16 = ((Bf & 48)/16 == 2)?((Bf & 48)/16):(((Bf3 & 3)/3)*3 + (Bf3 & 12)/12);  
            int t64 = ((Bf & 192)/64 == 1)?((Bf & 192)/64):(((Bf4 & 3)/3)*2 + ((Bf4 & 12)/12)*3);  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
}  
  
__global__ void UpdateButterflysFlippedH23Kernel( int* tiling, const int N, const int p1)  
{  
    // Finish updating adjacent slice in H21  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if((i%3) == (p1+1)%3 ) {  
            int Bf = tiling[i*N+j];  
            int s16 = ((Bf & 48)/16 != 2)?(1):(0);  
            int s64 = ((Bf & 192)/64 != 1)?(1):(0);  
            int Bf3 = tiling[(i+1)*N+j];  
            int Bf4 = tiling[(i+1)*N+j-1];  
              
            int c = (Bf & 256)/256;  
            int t1 = (Bf & 3);  
            int t4 = (Bf & 12)/4;  
            int t16 = ((Bf3 & 256) == 256)?(2):(s16*((Bf & 48)/16));;  
            int t64 = ((Bf4 & 256) == 256)?(1):(s64*((Bf & 192)/64));;  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
}  
  
__global__ void UpdateButterflysFlippedL1Kernel( int* tiling, const int N, const int p1, const int p2)  
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if((j%3) == p1 && (i&1) != p2)  
        {  
            int Bf1 = tiling[(i-1)*N+j];  
            int Bf2 = tiling[(i+1)*N+j];  
            int c = (tiling[i*N+j] & 256)/256;  
            int t1 = ((Bf1 & 12)/4 == 0)?(0):(((Bf1 & 12)/4)%3  + 1);  
            int t4 = ((Bf2 & 3) == 0)?(0):(((Bf2 & 3)+1)%3  + 1);  
            int t16 = ((Bf2 & 192)/64 == 0)?(0):(((Bf2 & 192)/64)%3 + 1);  
            int t64 = ((Bf1 & 48)/16 == 0)?(0):(((Bf1 & 48)/16 + 1)%3 + 1);  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
      
}  
  
__global__ void UpdateButterflysFlippedL21Kernel( int* tiling, const int N, const int p1)  
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if((j%3) == (p1+2)%3) { //right  
            int Bf = tiling[i*N+j];  
            int Bf1 = tiling[(i-1)*N+j+1];  
            int Bf2 = tiling[i*N+j+1];  
              
            int c = ((Bf1 & 48)/16 == 2)?(1):(0);  
            int t1 = ((Bf & 3) == 2)?((Bf & 3)):((Bf1 & 192)/192 + ((Bf1 & 48)/48)*3);  
            int t4 = ((Bf & 12)/4 == 1)?((Bf&12)/4):(((Bf2 & 192)/192)*3 + ((Bf2 & 48)/48)*2);  
            int t16 = (Bf & 48)/16;  
            int t64 = (Bf & 192)/64;  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
      
}  
  
__global__ void UpdateButterflysFlippedL22Kernel( int* tiling, const int N, const int p1)  
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if((j%3) == (p1+1)%3) { //left  
            int Bf = tiling[i*N+j];  
            int s1 = ((Bf & 3) != 2)?(1):(0);  
            int s4 = ((Bf & 12)/4 != 1)?(1):(0);  
              
            int Bf1 = tiling[(i-1)*N+j+1];  
            int Bf2 = tiling[i*N+j+1];  
            int Bf3 = tiling[(i+1)*N+j-1];  
            int Bf4 = tiling[i*N+j-1];  
              
            int c = ((Bf3 & 3) == 2)?(1):(0);  
            int t1 = ((Bf1 & 256)/256 == 1)?(2):(s1*(Bf & 3));  
            int t4 = ((Bf2 & 256)/256 == 1)?(1):(s4*(Bf & 12)/4);  
            int t16 = ((Bf & 48)/16 == 2)?((Bf & 48)/16):(((Bf3 & 3)/3)*3 + (Bf3 & 12)/12);  
            int t64 = ((Bf & 192)/64 == 1)?((Bf & 192)/64):(((Bf4 & 3)/3)*2 + ((Bf4 & 12)/12)*3);  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
      
}  
  
__global__ void UpdateButterflysFlippedL23Kernel( int* tiling, const int N, const int p1)  
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if((j%3) == (p1+2)%3) { //right  
            int Bf = tiling[i*N+j];  
            int s16 = ((Bf & 48)/16 != 2)?(1):(0);  
            int s64 = ((Bf & 192)/64 != 1)?(1):(0);  
              
            int Bf3 = tiling[(i+1)*N+j-1];  
            int Bf4 = tiling[i*N+j-1];  
              
            int c = (Bf & 256)/256;  
            int t1 = (Bf & 3);  
            int t4 = (Bf & 12)/4;  
            int t16 = ((Bf3 & 256)/256 == 1)?(2):(s16*(Bf & 48)/16);  
            int t64 = ((Bf4 & 256)/256 == 1)?(1):(s64*(Bf & 192)/64);  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
      
}  
  
__global__ void UpdateButterflysFlippedR1Kernel( int* tiling, const int N, const int p1, const int p2)  
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if(((i + j)%3) == p1 && (j&1) != p2) {  
            int Bf1 = tiling[(i+1)*N+j-1];  
            int Bf2 = tiling[(i-1)*N+j+1];  
              
            int c = (tiling[i*N+j] & 256)/256;  
            int t1 = ((Bf2 & 192)/64 == 0)?(0):(((Bf2 & 192)/64 + 1)%3 + 1);  
            int t4 = ((Bf2 & 48)/16 == 0)?(0):(((Bf2 & 48)/16)%3 + 1);  
            int t16 = ((Bf1 & 12)/4 == 0)?(0):(((Bf1 & 12)/4 + 1)%3 + 1);  
            int t64 = ((Bf1 & 3) == 0)?(0):(((Bf1 & 3))%3+1);  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
}  
  
__global__ void UpdateButterflysFlippedR21Kernel( int* tiling, const int N, const int p1)  
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if(((i + j)%3) == (p1+2)%3) { //right  
            int Bf = tiling[i*N+j];  
            int Bf2 = tiling[i*N+j+1];  
            int Bf3 = tiling[(i+1)*N+j];  
              
            int c = ((Bf3 & 3) == 1)?(1):(0);  
            int t1 = (Bf & 3);  
            int t4 = ((Bf & 12)/4 == 2)?((Bf & 12)/4):(((Bf2 & 3)/3) + ((Bf2 & 192)/192)*3);  
            int t16 = ((Bf & 48)/16 == 1)?((Bf & 48)/16):(((Bf3 & 3)/3)*3 + ((Bf3 & 192)/192)*2);  
            int t64 = (Bf & 192)/64;  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
      
}  
  
__global__ void UpdateButterflysFlippedR22Kernel( int* tiling, const int N, const int p1)  
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if(((i + j)%3) == (p1+1)%3) { //left  
            int Bf = tiling[i*N+j];  
            int s4 = ((Bf & 12)/4 != 2)?(1):(0);  
            int s16 = ((Bf & 48)/16 != 1)?(1):(0);  
              
            int Bf1 = tiling[(i-1)*N+j];  
            int Bf2 = tiling[i*N+j+1];  
            int Bf3 = tiling[(i+1)*N+j];  
            int Bf4 = tiling[i*N+j-1];  
              
            int c = ((Bf1 & 48)/16 == 1)?(1):(0);  
            int t1 = ((Bf & 3) == 1)?((Bf & 3)):(((Bf1 & 48)/48)*3 + ((Bf1 & 12)/12)*2);  
            int t4 = ((Bf2 & 256)/256 == 1)?(2):(s4*(Bf & 12)/4);  
            int t16 = ((Bf3 & 256)/256 == 1)?(1):(s16*(Bf & 48)/16);  
            int t64 = ((Bf & 192)/64 == 2)?((Bf & 192)/64):(((Bf4 & 48)/48) + ((Bf4 & 12)/12)*3);  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
      
}  
  
__global__ void UpdateButterflysFlippedR23Kernel( int* tiling, const int N, const int p1)  
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {  
        if(((i + j)%3) == (p1+2)%3) { //right  
            int Bf = tiling[i*N+j];  
            int s1 = ((Bf & 3) != 1)?(1):(0);  
            int s64 = ((Bf & 192)/64 != 2)?(1):(0);  
              
            int Bf1 = tiling[(i-1)*N+j];  
            int Bf4 = tiling[i*N+j-1];  
              
            int c = (Bf & 256)/256;  
            int t1 = ((Bf1 & 256)/256 == 1)?(1):(s1*(Bf & 3));  
            int t4 = (Bf & 12)/4;  
            int t16 = (Bf & 48)/16;  
            int t64 = ((Bf4 & 256)/256 == 1)?(2):(s64*(Bf & 192)/64);  
              
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;  
        }  
    }  
}  
  
  
__global__ void UpdateLozengeFromButterflysHKernel( int* tilingBH,  int* tilingLH, int* tilingLL, int* tilingLR, const int N)  
{  
    // update lozenges from horizontal butterfly  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {  
        int BF = tilingBH[i*N+j];  
        int lh1 = ((BF & 3)==3)?(1):(0);  
        int lh2 = ((BF & 12)==12)?(1):(0);  
        int lh4 = ((BF & 48)==48)?(1):(0);  
        int lh8 = ((BF & 192)==192)?(1):(0);  
        tilingLH[i*N+j] = lh1+2*lh2+4*lh4+8*lh8;  
          
        int ll1 = ((BF & 256)==256)?(1):(0);  
        int ll2 = ((BF & 48)==48)?(1):(0);  
        int ll4 = ((BF & 192)==64)?(1):(0);  
        int ll8 = ((BF & 192)==128)?(1):(0);  
        tilingLL[i*N+j] = ll1+2*ll2+4*ll4+8*ll8;  
          
        int lr1 = ((BF & 256)==256)?(1):(0);  
        int lr2 = ((BF & 48)==16)?(1):(0);  
        int lr4 = ((BF & 48)==32)?(1):(0);  
        int lr8 = ((BF & 192)==192)?(1):(0);  
        tilingLR[i*N+j] = lr1+2*lr2+4*lr4+8*lr8;  
    }  
      
}  
  
  
__global__ void UpdateLozengeFromButterflysLKernel( int* tilingBL, int* tilingLH, int* tilingLL, int* tilingLR, const int N)  
{  
    // update lozenges from left butterfly  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {  
        int BF = tilingBL[i*N+j];  
        int ll1 = ((BF & 3)==3)?(1):(0);  
        int ll2 = ((BF & 12)==12)?(1):(0);  
        int ll4 = ((BF & 48)==48)?(1):(0);  
        int ll8 = ((BF & 192)==192)?(1):(0);  
        tilingLL[i*N+j] = ll1+2*ll2+4*ll4+8*ll8;  
          
        int lr1 = ((BF & 3)==3)?(1):(0);  
        int lr2 = ((BF & 12)==4)?(1):(0);  
        int lr4 = ((BF & 12)==8)?(1):(0);  
        int lr8 = ((BF & 256)==256)?(1):(0);  
        tilingLR[i*N+j] = lr1+2*lr2+4*lr4+8*lr8;  
          
        int lh1 = ((BF & 3)==1)?(1):(0);  
        int lh2 = ((BF & 3)==2)?(1):(0);  
        int lh4 = ((BF & 12)==12)?(1):(0);  
        int lh8 = ((BF & 256)==256)?(1):(0);  
        tilingLH[i*N+j] = lh1+2*lh2+4*lh4+8*lh8;  
          
          
    }  
      
}  
  
__global__ void UpdateLozengeFromButterflysRKernel( int* tilingBR, int* tilingLH, int* tilingLL, int* tilingLR, const int N)  
{  
    // update lozenges from right butterfly  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
      
      
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {  
        int BF = tilingBR[i*N+j];  
        int lr1 = ((BF & 3)==3)?(1):(0);  
        int lr2 = ((BF & 12)==12)?(1):(0);  
        int lr4 = ((BF & 48)==48)?(1):(0);  
        int lr8 = ((BF & 192)==192)?(1):(0);  
        tilingLR[i*N+j] = lr1+2*lr2+4*lr4+8*lr8;  
          
        int lh1 = ((BF & 3)==1)?(1):(0);  
        int lh2 = ((BF & 3)==2)?(1):(0);  
        int lh4 = ((BF & 256)==256)?(1):(0);  
        int lh8 = ((BF & 192)==192)?(1):(0);  
        tilingLH[i*N+j] = lh1+2*lh2+4*lh4+8*lh8;  
          
        int ll1 = ((BF & 3)==3)?(1):(0);  
        int ll2 = ((BF & 256)==256)?(1):(0);  
        int ll4 = ((BF & 192)==64)?(1):(0);  
        int ll8 = ((BF & 192)==128)?(1):(0);  
        tilingLL[i*N+j] = ll1+2*ll2+4*ll4+8*ll8;  
    }  
      
}  
  
  
  
void RotateLozenges(dim3 block_size, dim3 grid_size,  curandStateMtgp32 * d_status,  int* tiling, const int N, const int t, const int c) 
{ 
RotateLozengesKernel << <grid_size, block_size >> > (d_status, tiling, N, t, c);
getLastCudaError("RotateLozengesKernel launch failed");
} 
 
void UpdateLozengesFlipped(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int t, const int c) 
{ 
UpdateLozengesFlippedKernel << <grid_size, block_size >> > (tiling, N, t, c);
getLastCudaError("UpdateLozengesFlippedKernel launch failed");
} 
 
void UpdateLozenges0(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2,  int* tiling3, const int N) 
{ 
UpdateLozenges0Kernel << <grid_size, block_size >> > (tiling1, tiling2, tiling3, N);
getLastCudaError("UpdateLozenges0Kernel launch failed");
} 
 
void UpdateLozenges1(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2,  int* tiling3, const int N) 
{ 
UpdateLozenges1Kernel << <grid_size, block_size >> > (tiling1, tiling2, tiling3, N);
getLastCudaError("UpdateLozenges1Kernel launch failed");
} 
 
void UpdateLozenges2(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2,  int* tiling3, const int N) 
{ 
UpdateLozenges2Kernel << <grid_size, block_size >> > (tiling1, tiling2, tiling3, N);
getLastCudaError("UpdateLozenges2Kernel launch failed");
} 
 
void UpdateLozenges(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2,  int* tiling3, const int N, const int t) 
{ 
UpdateLozengesKernel << <grid_size, block_size >> > (tiling1, tiling2, tiling3, N, t);
getLastCudaError("UpdateLozengesKernel launch failed");
} 
 
void UpdateTriangleUFromLozenges(dim3 block_size, dim3 grid_size,  int* tilingH,  int* tilingL,  int* tilingU, const int N) 
{ 
UpdateTriangleUFromLozengesKernel << <grid_size, block_size >> > (tilingH, tilingL, tilingU, N);
getLastCudaError("UpdateTriangleUFromLozengesKernel launch failed");
} 
 
void UpdateButterflysHFromLozenge(dim3 block_size, dim3 grid_size,  int* tilingBH,  int* tilingLH,  int* tilingLL,  int* tilingLR, const int N) 
{ 
UpdateButterflysHFromLozengeKernel << <grid_size, block_size >> > (tilingBH, tilingLH, tilingLL, tilingLR, N);
getLastCudaError("UpdateButterflysHFromLozengeKernel launch failed");
} 
 
void UpdateButterflysLFromLozenge(dim3 block_size, dim3 grid_size,  int* tilingBL,  int* tilingLH,  int* tilingLL,  int* tilingLR, const int N) 
{ 
UpdateButterflysLFromLozengeKernel << <grid_size, block_size >> > (tilingBL, tilingLH, tilingLL, tilingLR, N);
getLastCudaError("UpdateButterflysLFromLozengeKernel launch failed");
} 
 
void UpdateButterflysRFromLozenge(dim3 block_size, dim3 grid_size,  int* tilingBR,  int* tilingLH,  int* tilingLL,  int* tilingLR, const int N) 
{ 
UpdateButterflysRFromLozengeKernel << <grid_size, block_size >> > (tilingBR, tilingLH, tilingLL, tilingLR, N);
getLastCudaError("UpdateButterflysRFromLozengeKernel launch failed");
} 
 
void RotateTriangles(dim3 block_size, dim3 grid_size,  curandStateMtgp32 * d_status,  int* tiling, const int N, const int c) 
{ 
RotateTrianglesKernel << <grid_size, block_size >> > (d_status, tiling, N, c);
getLastCudaError("RotateTrianglesKernel launch failed");
} 
 
void UpdateTrianglesFlipped0(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int c) 
{ 
UpdateTrianglesFlipped0Kernel << <grid_size, block_size >> > (tiling, N, c);
getLastCudaError("UpdateTrianglesFlipped0Kernel launch failed");
} 
 
void UpdateTrianglesFlipped1(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int c) 
{ 
UpdateTrianglesFlipped1Kernel << <grid_size, block_size >> > (tiling, N, c);
getLastCudaError("UpdateTrianglesFlipped1Kernel launch failed");
} 
 
void UpdateTriangles(dim3 block_size, dim3 grid_size,  int* tiling1,  int* tiling2, const int N, const int t) 
{ 
UpdateTrianglesKernel << <grid_size, block_size >> > (tiling1, tiling2, N, t);
getLastCudaError("UpdateTrianglesKernel launch failed");
} 
 
void UpdateLozengeHFromTriangles(dim3 block_size, dim3 grid_size,  int* tilingU,  int* tilingD,  int* tilingH, const int N) 
{ 
UpdateLozengeHFromTrianglesKernel << <grid_size, block_size >> > (tilingU, tilingD, tilingH, N);
getLastCudaError("UpdateLozengeHFromTrianglesKernel launch failed");
} 
 
void UpdateLozengeLFromTriangles(dim3 block_size, dim3 grid_size,  int* tilingU,  int* tilingD,  int* tilingL, const int N) 
{ 
UpdateLozengeLFromTrianglesKernel << <grid_size, block_size >> > (tilingU, tilingD, tilingL, N);
getLastCudaError("UpdateLozengeLFromTrianglesKernel launch failed");
} 
 
void UpdateLozengeRFromTriangles(dim3 block_size, dim3 grid_size,  int* tilingU,  int* tilingD,  int* tilingR, const int N) 
{ 
UpdateLozengeRFromTrianglesKernel << <grid_size, block_size >> > (tilingU, tilingD, tilingR, N);
getLastCudaError("UpdateLozengeRFromTrianglesKernel launch failed");
} 
 
void RotateButterflys(dim3 block_size, dim3 grid_size,  curandStateMtgp32 * d_status,  int* tiling, const int N, const int t, const int p1, const int p2) 
{ 
RotateButterflysKernel << <grid_size, block_size >> > (d_status, tiling, N, t, p1, p2);
getLastCudaError("RotateButterflysKernel launch failed");
} 
 
void UpdateButterflysFlippedH1(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1, const int p2) 
{ 
UpdateButterflysFlippedH1Kernel << <grid_size, block_size >> > (tiling, N, p1, p2);
getLastCudaError("UpdateButterflysFlippedH1Kernel launch failed");
} 
 
void UpdateButterflysFlippedH21(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedH21Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedH21Kernel launch failed");
} 
 
void UpdateButterflysFlippedH22(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedH22Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedH22Kernel launch failed");
} 
 
void UpdateButterflysFlippedH23(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedH23Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedH23Kernel launch failed");
} 
 
void UpdateButterflysFlippedL1(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1, const int p2) 
{ 
UpdateButterflysFlippedL1Kernel << <grid_size, block_size >> > (tiling, N, p1, p2);
getLastCudaError("UpdateButterflysFlippedL1Kernel launch failed");
} 
 
void UpdateButterflysFlippedL21(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedL21Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedL21Kernel launch failed");
} 
 
void UpdateButterflysFlippedL22(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedL22Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedL22Kernel launch failed");
} 
 
void UpdateButterflysFlippedL23(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedL23Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedL23Kernel launch failed");
} 
 
void UpdateButterflysFlippedR1(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1, const int p2) 
{ 
UpdateButterflysFlippedR1Kernel << <grid_size, block_size >> > (tiling, N, p1, p2);
getLastCudaError("UpdateButterflysFlippedR1Kernel launch failed");
} 
 
void UpdateButterflysFlippedR21(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedR21Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedR21Kernel launch failed");
} 
 
void UpdateButterflysFlippedR22(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedR22Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedR22Kernel launch failed");
} 
 
void UpdateButterflysFlippedR23(dim3 block_size, dim3 grid_size,  int* tiling, const int N, const int p1) 
{ 
UpdateButterflysFlippedR23Kernel << <grid_size, block_size >> > (tiling, N, p1);
getLastCudaError("UpdateButterflysFlippedR23Kernel launch failed");
} 
 
void UpdateLozengeFromButterflysH(dim3 block_size, dim3 grid_size,  int* tilingBH,  int* tilingLH, int* tilingLL, int* tilingLR, const int N) 
{ 
UpdateLozengeFromButterflysHKernel << <grid_size, block_size >> > (tilingBH, tilingLH, tilingLL, tilingLR, N);
getLastCudaError("UpdateLozengeFromButterflysHKernel launch failed");
} 
 
void UpdateLozengeFromButterflysL(dim3 block_size, dim3 grid_size,  int* tilingBL, int* tilingLH, int* tilingLL, int* tilingLR, const int N) 
{ 
UpdateLozengeFromButterflysLKernel << <grid_size, block_size >> > (tilingBL, tilingLH, tilingLL, tilingLR, N);
getLastCudaError("UpdateLozengeFromButterflysLKernel launch failed");
} 
 
void UpdateLozengeFromButterflysR(dim3 block_size, dim3 grid_size,  int* tilingBR, int* tilingLH, int* tilingLL, int* tilingLR, const int N) 
{ 
UpdateLozengeFromButterflysRKernel << <grid_size, block_size >> > (tilingBR, tilingLH, tilingLL, tilingLR, N);
getLastCudaError("UpdateLozengeFromButterflysRKernel launch failed");
} 
 

