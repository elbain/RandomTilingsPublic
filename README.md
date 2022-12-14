# Random Tilings with the GPU
<img align="right" width = "200" src="https://github.com/LittleBadger/RandomTilings/blob/master/TriangleTiling.svg">
Here is an C++/OpenCL library* for generating random tilings efficiently with Markov Chain Monte Carlo on the GPU. See the companion paper [1] for further details.  At the moment, the library supports domino tilings, lozenge tilings, bibone tilings (dimers on the triangular lattice), rectangle-triangle tilings, and the six vertex model. The program also includes utility functions for constructing domains, maximal/minimal tilings, height functions, Maya diagrams/lattice paths/other representations, and for drawing pictures with Scalable Vector Graphics.

*C++/CUDA is now also supported.

## Building and Running

### Prerequisites
In order to build and run the program, you will need the following:
* [OpenCL](http://www.khronos.org/opencl)
* [TinyMT](https://github.com/MersenneTwister-Lab/TinyMT): A library for generating random numbers on the GPU, see [here](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/).
* [TinyMT Parameters](https://github.com/jj1bdx/tinymtdc-longbatch): A list of precomputed parameters for the TinyMT number generator.

or
* NVIDIA CUDA toolkit

### Compiling the examples
A sample makefile is included in the root directory, and a set minimal example tiling programs are in the folder /Examples to get you started. To build an example, e.g. MinimalDomino example, run from the root directory:
```
make MinimalDominoEx
```
or if using CUDA:
```
make MinimalDominoEx cuda=1
```
Running the compiled program will write its output to the folder Examples/ExampleOuts/MinimalDomino/.

### Usage
For details on how to specify domains/tilings and run the simulation, please see the comments in the header file for each model, along with the example programs in src/examples.

## Comments/Questions?
We would be happy to hear any comments or questions! Please email David (dkeating@berkeley.edu) or Ananth (asridhar@berkeley.edu). 

For CUDA-specific comments or questions, please email Emily (emily_bain@berkeley.edu).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## More Information
[1] D. Keating, A. Sridhar. "Random Tilings with the GPU." https://arxiv.org/pdf/1804.07250.pdf
