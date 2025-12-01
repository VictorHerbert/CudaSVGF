#include "utils.h"

#include <cuda_runtime.h>

void printGPUProperties(){
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0f << " KB" << std::endl;
    std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024.0f << " KB" << std::endl;

    double memClock = prop.memoryClockRate * 1000.0;
    double busWidthBytes = prop.memoryBusWidth / 8.0;
    double bandwidthGBs = (memClock * 2 * busWidthBytes) / 1e9;

    double smCount = prop.multiProcessorCount;
    double coresPerSM = 64.0;
    double totalCores = smCount * coresPerSM;
    double fp32TFLOPS = totalCores * prop.clockRate * 1000.0 * 2.0 / 1e12;

    std::cout << "Memory bandwidth: " << bandwidthGBs << " GB/s" << std::endl;
    std::cout << "Theoretical FP32 compute: " << fp32TFLOPS << " TFLOPS" << std::endl;
}
