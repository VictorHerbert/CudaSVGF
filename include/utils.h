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

void CHECK_CUDA(cudaError_t result);

void printGPUProperties();

#endif