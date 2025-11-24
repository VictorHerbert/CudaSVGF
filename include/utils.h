#pragma once
#ifndef UTILS_H
#define UTILS_h

#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#define KERNEL __global__
#define CUDA_FUNC __device__
#define CUDA_CPU_FUNC __device__ __host__
#define LAUNCHER


typedef unsigned char byte;

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        throw std::runtime_error(std::string("CUDA error: ") + errStr + " at line " + std::to_string(__LINE__)); \
    } \
} while(0)

void printGPUProperties();

#endif