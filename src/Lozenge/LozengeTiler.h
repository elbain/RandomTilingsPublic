
#ifndef LOZENGE_LOZENGETILER_H_
#define LOZENGE_LOZENGETILER_H_

#include "../common/common.h"
#ifdef __NVCC__
#include <curand_mtgp32.h>
#endif

// The domain of a lozenge tiling is a union of triangles on the triangular lattice.
// The domain is specified by a 2*N^2 dimension arrays of 0 and 1. See example for details.
//
// Tiling is stored on vertices of the triangular lattice. There is an indicator on each
// adjacent edge which is nonzero if a lozenge (or equivalently a dimer on the dual graph) crosses the edge, 
// and otherwise has a value depending on the edge is:
//        1   2
//     4 __\./__ 8
//         / \
//       16   32
// The tile state at the vertex is the sum of the indicators.
//
// The tiling is then tricolored such that all vertices of a given color can 'flip'
// without effecting each other. At each step in the Markov chain, it will first select a color
// then the gpu will attempt to flip all vertices of that color, then a seperate kernel will update
// the other colors based on the result of the flips (see lozengekernel.cl).
//
// Some technical notes:
// In order for dividing the tiling to work correctly it is neccessary that tiling is a square array with size divisible by 3. That is the size of tiling must be N*N, with N divisible by 3.
//
// In order for the update kernel to work correctly, the vector for each color must be padded by zeros; that is, the first and last rows and first and last columns must be zero. This is accomplished by making sure the tiling has 3 times the padding (first three and last three rows/cols are zero).
//
// Here the domain created by LozengeTiler::AlmostHexDomain results in a tiling that satisfies the above conditions.




class LozengeTiler {

private:
#ifndef __NVCC__
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl::make_kernel<cl::Buffer, const int> InitTinyMT;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int> RotateTiles;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, const int, const int> UpdateTiles;
	cl::Buffer tinymtparams;
#else
    mtgp32_kernel_params* devKernelParams;
    curandStateMtgp32* devMTGPStates;
#endif

public:
#ifndef __NVCC__
    LozengeTiler(cl::Context context0, cl::CommandQueue queue0, std::vector<cl::Device> devices, std::string source, cl_int &err) :
    	context(context0),
		queue(queue0),
    	program(LoadCLProgram(context,devices,source)),
		InitTinyMT(program, "InitTinyMT"),
		RotateTiles(program, "RotateTiles"),
		UpdateTiles(program, "UpdateTiles") { };
#else
    LozengeTiler() = default;
#endif

	// All of the following functions are almost identical to Domino.
	// Take a look at DominoTiler.h.
	
#ifndef __NVCC__
    void LoadTinyMT(std::string params, int size);
#else 
    void LoadMTGP();
#endif

    void Walk(tiling &t, int samples, long seed);
	
    static domain Hexagon(int N);

    static tiling MaxTiling(const domain &d);
    static tiling MinTiling(const domain &d);

    static tiling HeightfuncToTiling(const heightfunc &hf, const domain &d);
    static heightfunc TilingToHeightfunc(const tiling &t, const domain  &d);
    static std::vector<int> DomainToVertices(const domain &d);
    static domain TilingToDomain(const tiling &t);
    static domain AlmostHexDomain(int N, int M);

    static void DomainToSVG(const domain &d, std::string filename);
    static void DimerToSVG(const tiling &t, std::string filename);
    static void DimerToSVG(const tiling &t, const domain &d, std::string filename);
    static void TilingToSVG(const tiling &t, std::string filename);
    static void PrintDomain(const domain &d, std::string filename);
};

#endif /* LOZENGE_LOZENGETILER_H_ */
