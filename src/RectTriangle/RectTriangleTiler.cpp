//
//  RectTriangleTiler.cpp
//  
//
//  Created by Ananth, David
//

#include "RectTriangleTiler.h"
#include "../common/common.h"
#ifndef __NVCC__
#include "../TinyMT/file_reader.h"
#else
#include <cuda_runtime.h>
#include <curand.h>
#include "recttrianglekernel.cuh"
#include "../common/helper_cuda.h"
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#endif


#ifndef __NVCC__
void RectTriangleTiler::LoadTinyMT(std::string params, int size) {
    tinymtparams = get_params_buffer(params, context, queue, size);
}
#else
void RectTriangleTiler::LoadMTGP() {
    checkCudaErrors(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));
    checkCudaErrors(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));
}
#endif

void RectTriangleTiler::Walk(tiling &t, int steps, long seed) {
    int N = std::sqrt(t.size());
    
    std::vector<int> h_vR((N/3)*N,0); // initialize vectors
    std::vector<int> h_vB((N/3)*N,0); // initialize vectors
    std::vector<int> h_vG((N/3)*N,0); // initialize vectors
    
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            if(((i-j)%3+3)%3==0) {
                h_vR[i*(N/3)+(j/3)] = t[i*N+j];
            } else if(((i-j)%3+3)%3==1) {
                h_vB[i*(N/3)+(j/3)] = t[i*N+j];
            } else {
                h_vG[i*(N/3)+(j/3)] = t[i*N+j];
            }
        }
    }
    
#ifndef __NVCC__
    // device objects
    cl::Buffer d_vR = cl::Buffer(context, h_vR.begin(), h_vR.end(),CL_FALSE);
    cl::Buffer d_vB = cl::Buffer(context,h_vB.begin(), h_vB.end(),CL_FALSE);
    cl::Buffer d_vG = cl::Buffer(context,h_vG.begin(), h_vG.end(),CL_FALSE);
#else
    int* d_vR;
    int* d_vB;
    int* d_vG;
    checkCudaErrors(cudaMalloc((void**)&d_vR, (N / 3) * N * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_vR, h_vR.data(), (N / 3) * N * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&d_vB, (N / 3) * N * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_vB, h_vB.data(), (N / 3) * N * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&d_vG, (N / 3) * N * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_vG, h_vG.data(), (N / 3) * N * sizeof(int), cudaMemcpyHostToDevice));
#endif
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,2);
    generator.seed(seed);
    
#ifndef __NVCC__
    InitTinyMT( cl::EnqueueArgs(queue, cl::NDRange(N*N/3)), tinymtparams, seed );
#else
    int total_dim_x = N;
    int total_dim_y = N/3;
    int grid_x = (total_dim_x + BLOCK_DIM - 1) / BLOCK_DIM;
    int grid_y = (total_dim_y + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 block_size = dim3(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_size = dim3(grid_x, grid_y, 1);
    int num_states = grid_x * grid_y;
    checkCudaErrors(cudaMalloc((void**)&devMTGPStates, num_states * sizeof(curandStateMtgp32)));
    for (int i = 0; i < num_states; i++) {
        checkCudaErrors(curandMakeMTGP32KernelState(devMTGPStates + i, mtgp32dc_params_fast_11213, devKernelParams, 1, seed + i));
    }

#endif
    
    int r;
    for(int i = 0; i < steps; ++i) {
        r = distribution(generator);
        //r = 0;
#ifndef __NVCC__
        if (r%3==0) {
            flipTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), tinymtparams, d_vR, N);
            updateTiles1( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vR, d_vB, N, 0);
            updateTiles2( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vR, d_vG, N, 0);
        } else if (r%3==1) {
            flipTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), tinymtparams, d_vB, N);
            updateTiles1( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vB, d_vG, N, 1);
            updateTiles2( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vB, d_vR, N, 1);
        } else {
            flipTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), tinymtparams, d_vG, N);
            updateTiles1( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vG, d_vR, N, 2);
            updateTiles2( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vG, d_vB, N, 2);
        }
        if (!(i % 100000) && i != 0) {
            queue.finish();
            std::cout << "Walk step " << i << std::endl;
        }
        else if (!(i % 10000) && i != 0) {
            queue.flush();
        }
#else
        if (r%3 == 0) {
            flipTiles(block_size, grid_size, devMTGPStates, d_vR, N);
            updateTiles1(block_size, grid_size, d_vR, d_vB, N, 0);
            updateTiles2(block_size, grid_size, d_vR, d_vG, N, 0);
        }
        else if (r%3 == 1) {
            flipTiles(block_size, grid_size, devMTGPStates, d_vB, N);
            updateTiles1(block_size, grid_size, d_vB, d_vG, N, 1);
            updateTiles2(block_size, grid_size, d_vB, d_vR, N, 1);
        }
        else {
            flipTiles(block_size, grid_size, devMTGPStates, d_vG, N);
            updateTiles1(block_size, grid_size, d_vG, d_vR, N, 2);
            updateTiles2(block_size, grid_size, d_vG, d_vB, N, 2);
        }
#endif
    }
    
    // get memory from device
#ifndef __NVCC__
    cl::copy(queue, d_vR, h_vR.begin(), h_vR.end());
    cl::copy(queue, d_vB, h_vB.begin(), h_vB.end());
    cl::copy(queue, d_vG, h_vG.begin(), h_vG.end());
#else
    checkCudaErrors(cudaMemcpy(h_vR.data(), d_vR, (N / 3)* N * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_vB.data(), d_vB, (N / 3)* N * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_vG.data(), d_vG, (N / 3)* N * sizeof(int), cudaMemcpyDeviceToHost));
#endif
    
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            if(((i-j)%3+3)%3==0) {
               t[i*N+j] = h_vR[i*(N/3)+(j/3)];
            } else if(((i-j)%3+3)%3==1) {
                t[i*N+j] = h_vB[i*(N/3)+(j/3)];
            } else {
                t[i*N+j] = h_vG[i*(N/3)+(j/3)];
            }
        }
    }
    
//    std::cout<<"Tiling:"<<std::endl;
//    for(int i=0; i<N; ++i) {
//        for(int j=0; j<N; ++j) {
//            std::cout<<std::hex<<t[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::dec<<std::endl;
#ifdef __NVCC__
    checkCudaErrors(cudaFree(d_vR));
    checkCudaErrors(cudaFree(d_vB));
    checkCudaErrors(cudaFree(d_vG));
    checkCudaErrors(cudaFree(devMTGPStates));
#endif
}

tiling RectTriangleTiler::slopeHex(int N) {
    int M = 2*N+1; // tile vectors should be M*M in size
    
    std::vector<int> slopeTiling(M*M,0);
    
    
    for(int i=1; i<M-1; ++i) { // sloped tiling
        for(int j=1; j<M-1; ++j) {
            if (i == N && j == N) { // center
                if (N%2 == 0) {
                    slopeTiling[i*M+j] = 0x213312;
                } else {
                    slopeTiling[i*M+j] = 0x122133;
                }
            } else if (i == N) { // boundary
                if (j<N-1) {
                    if (j%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x1be1fc;
                    } else {
                        slopeTiling[i*M+j] = 0xe81c91;
                    }
                } else if (j==N-1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x122133;
                    } else {
                        slopeTiling[i*M+j] = 0xe81c91;
                    }
                } else if (j>N+1) {
                    if (j%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0xcf1eb1;
                    } else {
                        slopeTiling[i*M+j] = 0x19c18e;
                    }
                } else if (j==N+1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] =0x331221;
                    } else {
                        slopeTiling[i*M+j] =0x29c38e;
                    }
                }
            } else if (j == N) { // boundary
                if (i<N-1) {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0xd22afc;
                    } else {
                        slopeTiling[i*M+j] = 0x7af229;
                    }
                } else if (i==N-1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x122133;
                    } else {
                        slopeTiling[i*M+j] = 0x7af229;
                    }
                } else if (i > N+1) {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0xcfa22d;
                    } else {
                        slopeTiling[i*M+j] = 0x922fa7;
                    }
                } else if (i==N+1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] =0x331221;
                    } else {
                        slopeTiling[i*M+j] = 0x913fa7;
                    }
                }
            } else if (i==j) { //boundary
                if (i<N-1) {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x33aebd;
                    } else {
                        slopeTiling[i*M+j] = 0xbd7833;
                    }
                } else if (i > N+1) {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0xdbea33;
                    } else {
                        slopeTiling[i*M+j] = 0x3387db;
                    }
                } else if (i==N+1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x122133;
                    } else {
                        slopeTiling[i*M+j] = 0x3387db;
                    }
                } else if (i==N-1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x331221;
                    } else {
                        slopeTiling[i*M+j] = 0xbd7812;
                    }
                }
            } else if (i<N && j<i) { // slice
                if (i==N-1 && j<i-1) {
                    if (j%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x855be8;
                    } else {
                        slopeTiling[i*M+j] = 0x58881b;
                    }
                } else if (i<N-1 && j==i-1) {
                    if (j%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x8eb558;
                    } else {
                        slopeTiling[i*M+j] = 0x58388e;
                    }
                } else if (i==N-1 && j==i-1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x583812;
                    } else {
                        slopeTiling[i*M+j] = 0x8ebbe8;
                    }
                } else {
                    if (j%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x855558;
                    } else {
                        slopeTiling[i*M+j] = 0x588885;
                    }
                }
            } else if (i>N && j>i) { // slice
                if (i==N+1 && j>i+1) {
                    if (j%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x8eb558;
                    } else {
                        slopeTiling[i*M+j] = 0xb18885;
                    }
                } else if (i>N+1 && j==i+1) {
                    if (j%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x855be8;
                    } else {
                        slopeTiling[i*M+j] = 0xe88385;
                    }
                } else if (i==N+1 && j==i+1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x218385;
                    } else {
                        slopeTiling[i*M+j] = 0x8ebbe8;
                    }
                } else {
                    if (j%2 ==1) { //maybe
                        slopeTiling[i*M+j] = 0x855558;
                    } else {
                        slopeTiling[i*M+j] = 0x588885;
                    }
                }
            } else if (j<N && /*j>1 &&*/ i<j ) { // slice //maybe
                if (j==N-1 && i<j-1) {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x47d47a;
                    } else {
                        slopeTiling[i*M+j] = 0x7477d2;
                    }
                } else if (j<N-1 && i==j-1) {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0xa74d74;
                    } else {
                        slopeTiling[i*M+j] = 0x7473a7;
                    }
                } else if (j==N-1 && i==j-1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x747312;
                    } else {
                        slopeTiling[i*M+j] = 0xa7dd7a;
                    }
                } else {
                    if (i%2 ==1) { //maybe
                        slopeTiling[i*M+j] = 0x474474;
                    } else {
                        slopeTiling[i*M+j] = 0x747747;
                    }
                }
            } else if (j>N && i>j) { // slice
                if (j==N+1 && i>j+1) {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0xa74d74;
                    } else {
                        slopeTiling[i*M+j] = 0x2d7747;
                    }
                } else if (j>N+1 && i==j+1) {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x47d47a;
                    } else {
                        slopeTiling[i*M+j] = 0x7a3747;
                    }
                } else if (j==N+1 && i==j+1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x213747;
                    } else {
                        slopeTiling[i*M+j] = 0xa7dd7a;
                    }
                } else {
                    if (i%2 == 1) { //maybe
                        slopeTiling[i*M+j] = 0x474474;
                    } else {
                        slopeTiling[i*M+j] = 0x747747;
                    }
                }
            } else if (i<N && j>N && i>j-N) { // slice //maybe
                if (i==N-1 && j>N+1) {
                    if ((i+j)%2 != N%2) {
                        slopeTiling[i*M+j] = 0x6699cf;
                    } else {
                        slopeTiling[i*M+j] = 0x996f19;
                    }
                } else if (i<N-1 && j==N+1) {
                    if ((i+j)%2 != N%2) {
                        slopeTiling[i*M+j] = 0xfc9966;
                    } else {
                        slopeTiling[i*M+j] = 0x296c99;
                    }
                } else if (i==N-1 && j==N+1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x296319;
                    } else {
                        slopeTiling[i*M+j] = 0xfc99cf;
                    }
                } else {
                    if ((i+j)%2 != N%2) {
                        slopeTiling[i*M+j] = 0x669966;
                    } else {
                        slopeTiling[i*M+j] = 0x996699;
                    }
                }
                
            } else if (i>N && j<N && i<j+N) { // slice
                if (i==N+1 && j<N-1) {
                    if ((i+j)%2 != N%2) {
                        slopeTiling[i*M+j] = 0xfc9966;
                    } else {
                        slopeTiling[i*M+j] = 0x91f699;
                    }
                } else if (i>N+1 && j==N-1) {
                    if ((i+j)%2 != N%2) {
                        slopeTiling[i*M+j] = 0x6699cf;
                    } else {
                        slopeTiling[i*M+j] = 0x99c692;
                    }
                } else if (i==N+1 && j==N-1) {
                    if (N%2 == 0) {
                        slopeTiling[i*M+j] = 0x913692;
                    } else {
                        slopeTiling[i*M+j] = 0xfc99cf;
                    }
                } else {
                    if ((i+j)%2 != N%2) {
                        slopeTiling[i*M+j] = 0x669966;
                    } else {
                        slopeTiling[i*M+j] = 0x996699;
                    }
                }
            } else {
                slopeTiling[i*M+j] = 0x0;
            }
        }
    }
    for(int i=2; i<N-1; ++i) {
        slopeTiling[i*M] = 0x008085; //top left
        slopeTiling[i] = 0x000747; //top
    }
    slopeTiling[M] = 0x00308e;
    slopeTiling[1] = 0x0003a7;
    slopeTiling[(N-1)*M] = 0x00801b;
    slopeTiling[(N-1)] = 0x0007d2;
    for(int i=N+2; i<M-2; ++i) {
        slopeTiling[i*M+(M-1)] = 0x580800; //bottom right
        slopeTiling[(M-1)*M+i] = 0x747000; //bottom
    }
    slopeTiling[(N+1)*M+(M-1)] = 0xb10800;
    slopeTiling[(M-1)*M+(N+1)] = 0x2d7000;
    slopeTiling[(M-2)*M+(M-1)] = 0xe80300;
    slopeTiling[(M-1)*M+(M-2)] = 0x7a3000;
    for(int i=2; i<N-1; ++i) { // to do
        slopeTiling[i*M+i+N] = 0x900690; // top right
        slopeTiling[(i+N)*M+i] = 0x096009; // bottom left
    }
    slopeTiling[M+(N+1)] = 0x200c90;
    slopeTiling[(N+1)*M+1] = 0x01f009;
    slopeTiling[(N-1)*M+(M-2)] = 0x900f10;
    slopeTiling[(M-2)*M+(N-1)] = 0x09c002;
    if(N == 1) { //corners
        slopeTiling[0] = 0x000012;
        slopeTiling[(M-1)*M+N] = 0x013000;
        slopeTiling[N*M+(M-1)] = 0x200300;
        slopeTiling[N] = 0x000220;
        slopeTiling[N*M] = 0x001001;
        slopeTiling[(M-1)*M+(M-1)] = 0x330000;
    } else {
        slopeTiling[0] = 0x000033;
        slopeTiling[(M-1)*M+N] = 0x022000;
        slopeTiling[N*M+(M-1)] = 0x100100;
        slopeTiling[N] = 0x000220;
        slopeTiling[N*M] = 0x001001;
        slopeTiling[(M-1)*M+(M-1)] = 0x330000;
    }
    
    int M2 = M+6 + (3-M%3)%3;
    std::vector<int> slopeTiling2(M2*M2,0);//Change of coordinates
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int i2 = i+3;
            int j2 = j + N+3 - i;
            if ( j2 < M2 ) slopeTiling2[i2*M2+j2]  = slopeTiling[i*M+j];
        }
    }
    
    return slopeTiling2;
}

tiling RectTriangleTiler::maxHex(int N) {
    int M = 2*N+1; // tile vectors should be M*M in size
    
    std::vector<int> maxTiling(M*M,0);
    
    for(int i=1; i<M-1; ++i) { // Build highest height configuration
        for(int j=0; j<M; ++j) {
            if (j-i>N || i-j>N) {
                maxTiling[i*M+j] = 0x0;
            } else if(i<N && j<N) {
                maxTiling[i*M+j] = 0x333333;
            } else if(i-j<0 && j>N) {
                maxTiling[i*M+j] = 0x111111;
            } else if(i>N && i-j>0) {
                maxTiling[i*M+j] = 0x222222;
            } else if(i==N && j<N) {
                maxTiling[i*M+j] = 0x333222;
            } else if(i<N && j==N) {
                maxTiling[i*M+j] = 0x331311;
            } else if(i-j==0 && i>N) {
                maxTiling[i*M+j] = 0x211221;
            } else if(i==N && j==N) {
                maxTiling[i*M+j] = 0x331221;
            } else {
                std::cout<<"("<<i<<","<<j<<") FFFFUUUUUUUCCCCCCCKKKKKKK!!!!!!!!"<<std::endl;
            }
        }
    }
    for(int i=1; i<N; ++i) {
        maxTiling[i*M] = 0x003033; //top left
        maxTiling[i] = 0x000333; //top
    }
    for(int i=N+1; i<M-1; ++i) {
        maxTiling[i*M+(M-1)] = 0x110100; //bottom right
        maxTiling[(M-1)*M+i] = 0x222000; //bottom
    }
    for(int i=1; i<N; ++i) {
        maxTiling[i*M+i+N] = 0x100110; // top right
        maxTiling[(i+N)*M+i] = 0x022002; // bottom left
    }
    maxTiling[0] = 0x000033;
    maxTiling[N*M] = 0x003002;
    maxTiling[N] = 0x000310;
    maxTiling[(M-1)*M+(M-1)] = 0x210000;
    maxTiling[N*M+(M-1)] = 0x100100;
    maxTiling[(M-1)*M+N] = 0x022000;
    
    int M2 = M+6 + (3-M%3)%3;
    std::vector<int> maxTiling2(M2*M2,0);//Change of coordinates
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int i2 = i+3;
            int j2 = j + N+3 - i;
            if ( j2 < M2 ) maxTiling2[i2*M2+j2]  = maxTiling[i*M+j];
        }
    }
    
    return maxTiling2;
}

tiling RectTriangleTiler::minHex(int N) {
    int M = 2*N+1; // tile vectors should be M*M in size
    
    std::vector<int> minTiling(M*M,0);
    
    for(int i=1; i<M-1; ++i) { // Build min height configuration
        for(int j=1; j<M-1; ++j) {
            if (j-i>N || i-j>N) {
                minTiling[i*M+j] = 0;
            } else if(i>N && j>N) {
                minTiling[i*M+j] = 0x333333;// good
            } else if(i-j>0 && j<N) {
                minTiling[i*M+j] = 0x111111;// good
            } else if(i<N && i-j<0) {
                minTiling[i*M+j] = 0x222222;// good
            } else if(i==N && j>N) {
                minTiling[i*M+j] = 0x222333;// good
            } else if(i>N && j==N) {
                minTiling[i*M+j] = 0x113133;// good
            } else if(i-j==0 && i<N) {
                minTiling[i*M+j] = 0x122112;// good
            } else if(i==N && j==N) {
                minTiling[i*M+j] = 0x122133;// good
            } else {
                std::cout<<"("<<i<<","<<j<<") FFFFUUUUUUUCCCCCCCKKKKKKK!!!!!!!!"<<std::endl;
                exit(1);
            }
        }
    }
    for(int i=1; i<N; ++i) {
        minTiling[i*M] = 0x001011; //top left
        minTiling[i] = 0x000222; //top
    }
    for(int i=N+1; i<M-1; ++i) {
        minTiling[i*M+(M-1)] = 0x330300; //bottom right
        minTiling[(M-1)*M+i] = 0x333000; //bottom
    }
    for(int i=1; i<N; ++i) {
        minTiling[i*M+i+N] = 0x200220; // top right
        minTiling[(i+N)*M+i] = 0x011001; // bottom left
    }
    minTiling[0] = 0x000012;
    minTiling[(M-1)*M+N] = 0x013000;
    minTiling[N*M+(M-1)] = 0x200300;
    minTiling[N] = 0x000220;
    minTiling[N*M] = 0x001001;
    minTiling[(M-1)*M+(M-1)] = 0x330000;
    
    int M2 = M+6 + (3-M%3)%3;
    std::vector<int> minTiling2(M2*M2,0);//Change of coordinates
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int i2 = i+3;
            int j2 = j + N+3 - i;
            if ( j2 < M2 ) minTiling2[i2*M2+j2]  = minTiling[i*M+j];
        }
    }
    
    return minTiling2;
    
}

tiling RectTriangleTiler::LozengeToRectTriangle(tiling &tL, domain &dL) {
    int N = sqrt(tL.size());
    int M = sqrt(dL.size()/2);
    tiling tiling(N*N,0);
    
    for(int i=0; i<M; ++i) {
        for(int j=0; j<2*M; ++j) {
            if(dL[i*2*M+j] == 1) { // assumes necessary padding
                if(j%2 == 0) {
                    if((tL[i*N+j/2] & 8) == 8) {
                        tiling[i*N+j/2] += 1*0x1;
                        tiling[i*N+j/2+1] += 1*0x100;
                        tiling[(i+1)*N+j/2] += 1*0x10000;
                    } else if((tL[i*N+j/2] & 32) == 32) {
                        tiling[i*N+j/2] += 3*0x1;
                        tiling[i*N+j/2+1] += 3*0x100;
                        tiling[(i+1)*N+j/2] += 3*0x10000;
                    } else {
                        tiling[i*N+j/2] += 2*0x1;
                        tiling[i*N+j/2+1] += 2*0x100;
                        tiling[(i+1)*N+j/2] += 2*0x10000;
                    }
                } else {
                    if((tL[i*N+j/2+1] & 16) == 16) {
                        tiling[i*N+j/2+1] += 2*0x10;
                        tiling[(i+1)*N+j/2] += 2*0x1000;
                        tiling[(i+1)*N+j/2+1] += 2*0x100000;
                    } else if((tL[i*N+j/2+1] & 32) == 32) {
                        tiling[i*N+j/2+1] += 3*0x10;
                        tiling[(i+1)*N+j/2] += 3*0x1000;
                        tiling[(i+1)*N+j/2+1] += 3*0x100000;
                    } else {
                        tiling[i*N+j/2+1] += 1*0x10;
                        tiling[(i+1)*N+j/2] += 1*0x1000;
                        tiling[(i+1)*N+j/2+1] += 1*0x100000;
                    }
                }
            }
        }
    }
            
    return tiling;
}

void RectTriangleTiler::TilingToSVG(tiling &t, std::string filename) {
    std::ofstream outputFile(filename.c_str());
    
    int N = sqrt(t.size());
    int M = N-1;
    
    std::string BlackStyle = "\" style=\"fill:paleturquoise;stroke:black;stroke-width:0\" />\n";
    std::string WhiteStyle = "\" style=\"fill:deepskyblue;stroke:black;stroke-width:0\" />\n";
    
    double W = sqrt3*2*(N+1); double H = 3/2.0*(N+1);
    
    double s = 20;
    outputFile<< "<svg xmlns='http://www.w3.org/2000/svg' height=\"" <<s*H<< "\" width=\""<<s*W<<"\" >\n";
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            //std::cout<<i<<","<<j<<" "<<t[i*N+j]<<"\t";
            outputFile<< TilePicture((t[i*N+j+1] & (0xf << 4)) >> 4,i,2*j+1);
            outputFile<< TilePicture(t[i*N+j] & 0xf,i,2*j);
        }
        //std::cout<<std::endl;
    }
    
    outputFile<< "</svg>";
    outputFile.close();
}

double vx(int i, int j) {
    return 20*(i * 1/2.0 + j);
}

double vy(int i, int j) {
    return 20*(i * sqrt3/2.0) ;
}

std::string RectTriangleTiler::TilePicture(int TriType, int i, int j) {
    std::vector<std::string> colorStyles; colorStyles.push_back("");
    colorStyles.push_back("\" style=\"fill:midnightblue;stroke:midnightblue;stroke-width:.4\" />\n");
    colorStyles.push_back("\" style=\"fill:slategrey;stroke:slategrey;stroke-width:.4\" />\n");
    colorStyles.push_back("\" style=\"fill:lightsteelblue;stroke:lightsteelblue;stroke-width:.4\" />\n");
    
    std::string edgeStyle = " style=\"stroke:rgb(0,0,0);stroke-width:.5\" stroke-linecap = \"round\" />\n";
    
    // brute forced, probably a better way to arrange these
    
    std::stringstream pic;
    
    /*
     *            Q P
     *             R
     *
     *             R
     *            P Q
     */
    
    bool PQ=false, PR=false, QR=false, p=false, q=false, r=false;
    int PQR = 0, pPR = 0, pPQ = 0, rRQ = 0, rRP = 0, qQR = 0, qQP = 0;
    
    switch (TriType) {
        case 1: PR = 1; QR = 1; r = 1; PQR = 1; break;
        case 2: QR = 1; PQ = 1; q = 1; PQR = 3; break;
        case 3: PQ = 1; PR = 1; p = 1; PQR = 2; break;
        case 4: r = 1; PQR = 2; break;
        case 5: q = 1; PQR = 1; break;
        case 6: p = 1; PQR = 3; break;
        case 7: PQ = 1; PQR = 2; break;
        case 8: PR = 1; PQR = 1; break;
        case 9: QR = 1; PQR = 3; break;
        case 10: rRP = 1; rRQ = 2; r = 1; PR = 1; break;
        case 11: qQP = 1; qQR = 3; q = 1; QR = 1; break;
        case 12: pPQ = 2; pPR = 3; p = 1; PQ = 1; break;
        case 13: rRP = 2; rRQ = 1; r = 1; QR = 1; break;
        case 14: qQP = 3; qQR = 1; q = 1; PQ = 1; break;
        case 15: pPQ = 3; pPR = 2; p = 1; PR = 1; break;
    }
    
    // Remember:
    // The (i,j) face is adjacent to vertices:
    // if j is even: (i,j/2), (i+1,j/2), (i,j/2+1)
    // if j is odd: (i+1,j/2+1), (i+1,j/2), (i,j/2+1)
    
    double Px, Py, Qx, Qy, Rx, Ry;
    
    if ( j % 2 == 1 ) {
        Px = vx(i+1,j/2);   Py = vy(i+1,j/2);
        Qx = vx(i+1,j/2+1);   Qy = vy(i+1,j/2+1);
        Rx = vx(i,j/2+1);   Ry = vy(i,j/2+2);
    } else {
        Px = vx(i,j/2+1);   Py = vy(i,j/2+1);
        Qx = vx(i,j/2);   Qy = vy(i,j/2);
        Rx = vx(i+1,j/2);   Ry = vy(i+1,j/2);
    }
    
    double px = (Rx + Qx)/2; double py = (Ry + Qy)/2;
    double qx = (Px + Rx)/2; double qy = (Py + Ry)/2;
    double rx = (Px + Qx)/2; double ry = (Py + Qy)/2;
    
    
    if ( PQR > 0 ) { pic<< "<polygon points=\"" <<Px<<","<<Py<<" "<<Qx<<","<<Qy<<" "<<Rx<<","<<Ry<<" ";
        pic<< colorStyles[PQR]; }
    
    if ( pPR > 0 ) { pic<< "<polygon points=\"" <<Px<<","<<Py<<" "<<px<<","<<py<<" "<<Rx<<","<<Ry<<" ";
        pic<< colorStyles[pPR]; }
    
    if ( pPQ > 0 ) { pic<< "<polygon points=\"" <<Px<<","<<Py<<" "<<px<<","<<py<<" "<<Qx<<","<<Qy<<" ";
        pic<< colorStyles[pPQ];
    }
    if ( rRQ > 0 ) { pic<< "<polygon points=\"" <<Rx<<","<<Ry<<" "<<rx<<","<<ry<<" "<<Qx<<","<<Qy<<" ";
        pic<< colorStyles[rRQ];
    }
    
    if ( rRP > 0 ) { pic<< "<polygon points=\"" <<Rx<<","<<Ry<<" "<<rx<<","<<ry<<" "<<Px<<","<<Py<<" ";
        pic<< colorStyles[rRP];
    }
    if ( qQR > 0 ) { pic<< "<polygon points=\"" <<Qx<<","<<Qy<<" "<<qx<<","<<qy<<" "<<Rx<<","<<Ry<<" ";
        pic<< colorStyles[qQR];
    }
    if ( qQP > 0 ) { pic<< "<polygon points=\"" <<Qx<<","<<Qy<<" "<<qx<<","<<qy<<" "<<Px<<","<<Py<<" ";
        pic<< colorStyles[qQP];
    }
    
    if ( PQ ) pic << "<line x1=\"" <<Px<< "\" y1=\""<<Py<<"\" x2=\""<< Qx <<"\" y2=\""<< Qy<<"\""<< edgeStyle;
    if ( PR ) pic << "<line x1=\"" <<Px<< "\" y1=\""<<Py <<"\" x2=\""<< Rx <<"\" y2=\""<< Ry<<"\""<< edgeStyle;
    if ( QR ) pic << "<line x1=\"" <<Rx<< "\" y1=\""<<Ry<<"\" x2=\""<< Qx <<"\" y2=\""<< Qy<<"\""<< edgeStyle;
    
    if ( p ) pic << "<line x1=\"" <<Px<< "\" y1=\""<<Py<<"\" x2=\""<< px <<"\" y2=\""<< py<<"\""<< edgeStyle;
    if ( q ) pic << "<line x1=\"" <<Qx<< "\" y1=\""<<Qy<<"\" x2=\""<< qx <<"\" y2=\""<< qy<<"\""<< edgeStyle;
    if ( r ) pic << "<line x1=\"" <<Rx<< "\" y1=\""<<Ry<<"\" x2=\""<< rx <<"\" y2=\""<< ry<<"\""<< edgeStyle;
    
    return pic.str();
}
