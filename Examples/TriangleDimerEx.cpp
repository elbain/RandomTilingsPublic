/*
 *
 * Tiles an equilateral triangle of side length N, oriented point downward, with a equillateral triangle of size M cut out of the top left corner.  
 */

#include "../src/common/common.h"
#include "../src/TriangleDimer/TriangleDimerTiler.h"

int main() {

	int N = 6;

	int Nsteps = 500;
	
    std::cout<<"Running basic Dimer on Triangular Lattice example."<<std::endl;
    auto start = std::chrono::steady_clock::now();
    
#ifndef __NVCC__
    //Standard OpenCL set up code.
    //PrintOpenCLInfo(); // Look at what devices are available
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    std::string sinfo;
    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    devices[0].getInfo(CL_DEVICE_NAME, &sinfo);
    cl::CommandQueue queue(context);
    cl_int err = 0;
#endif
    
    //Create domain and starting tiling and draw them.
    tiling t = TriangleDimerTiler::IceCreamCone(N,0);
    domain d = TriangleDimerTiler::TilingToDomain(t);
    SaveMatrix(d,"./Examples/ExampleOuts/TriangleDimer/Domain.txt");
    SaveMatrix(t,"./Examples/ExampleOuts/TriangleDimer/DimerStart.txt");
    TriangleDimerTiler::DomainToSVG(d,"./Examples/ExampleOuts/TriangleDimer/Domain.svg");
    TriangleDimerTiler::DimerToSVG(t,"./Examples/ExampleOuts/TriangleDimer/DimerStart.svg");
    TriangleDimerTiler::TilingToSVG(t,"./Examples/ExampleOuts/TriangleDimer/TilingStart.svg");
    
    
    //Set up the mcmc.
#ifndef __NVCC__
    TriangleDimerTiler T(context, queue, devices, "./src/TriangleDimer/triangledimerkernel.cl", err);
    T.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", t.size());
#else 
// make the tiler
    TriangleDimerTiler T;

    // load the MTGP random number generators
    T.LoadMTGP();
    std::cout << "Loaded MTGP" << std::endl;
#endif 
    
    //Walk.
    std::random_device seed;
    T.Walk(t, Nsteps, seed());
    
    //Output final states.
    SaveMatrix(t,"./ExampleTilings/TriangleDimer/DimerEnd.txt");
    TriangleDimerTiler::DimerToSVG(t,"./Examples/ExampleOuts/TriangleDimer/DimerEnd.svg");
    TriangleDimerTiler::TilingToSVG(t,"./Examples/ExampleOuts/TriangleDimer/TilingEnd.svg");
    
    
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    typedef std::chrono::duration<float> float_seconds;
    auto secs = std::chrono::duration_cast<float_seconds>(diff);
    std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
}

