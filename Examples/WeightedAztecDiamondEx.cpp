/*
 * WeightedAztecDiamondEx.cpp
 *
 *
 * Tiles an Aztec Diamond of order N with two-periodic weights.
 */

#include "../src/common/common.h"
#include "../src/Domino/DominoTiler.h"

int main() {

	int N = 16;
	// weights are set in the kernel dominokerneltwoperiodic.cl or dominokerneltwoperiodic.cu

	int Nsteps = 5000;  

	std::cout << "Running Aztec Diamond of order " << N << "." << std::endl;

	auto start = std::chrono::steady_clock::now();

#ifndef __NVCC__
	//Standard OpenCL set up code.
	cl::Context context(CL_DEVICE_TYPE_DEFAULT);
	std::string sinfo;
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	devices[0].getInfo(CL_DEVICE_NAME, &sinfo); // Check which GPU you use!
	std::cout << "Created context using: " << sinfo << std::endl;
	cl::CommandQueue queue(context);

	cl_int err = 0;
#endif

	// Make the domain. 
	domain d = DominoTiler::AztecDiamond(N);

	// start with minimal tiling
	tiling t = DominoTiler::MinTiling(d);

	SaveMatrix(t, "./Examples/ExampleOuts/WeightedAztecDiamond/TilingStart.txt");
	DominoTiler::TilingToSVG(t, "./Examples/ExampleOuts/WeightedAztecDiamond/TilingStart.svg");
	std::cout << "Saved intial tiling" << std::endl;

#ifndef __NVCC__
	DominoTiler D(context, queue, devices, "./src/Domino/dominokerneltwoperiodic.cl", err); // use the domino kernel
	D.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", t.size() / 2);
#else 
	DominoTiler D;
	D.LoadMTGP();
	std::cout << "Loaded MTGP" << std::endl;
#endif 

	//Walk.
	std::random_device seed;
	std::cout << "Starting walk" << std::endl;

	D.Walk(t, Nsteps, seed());

	std::cout << "Finished!" << std::endl;

	//Output final states.
	SaveMatrix(t, "./Examples/ExampleOuts/WeightedAztecDiamond/TilingEnd.txt");
	DominoTiler::TilingToSVG(t, "./Examples/ExampleOuts/WeightedAztecDiamond/TilingEnd.svg");

	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	typedef std::chrono::duration<float> float_seconds;
	auto secs = std::chrono::duration_cast<float_seconds>(diff);
	std::cout << "Size: " << N << ".  Time elapsed: " << secs.count() << " s." << std::endl;

}

