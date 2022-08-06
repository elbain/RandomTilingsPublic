/*
 * cudaTest.cpp
 *
 * This program prints your CUDA information.
 */

#include "../src/common/common.h"
#include <cuda_runtime.h>

int main() {
	int n_devices;
	cudaGetDeviceCount(&n_devices);
	std::cout << "Number of CUDA devices: " << n_devices << std::endl;
	for (int i = 0; i < n_devices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "Device Number: " << i << std::endl;
		std::cout << "  Device name: " << prop.name << std::endl;
		std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
		std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
	}
}
