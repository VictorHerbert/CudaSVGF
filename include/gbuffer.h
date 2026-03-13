#ifndef GBUFFER_H
#define GBUFFER_H

#include "image.h"
#include "extended_math.h"
#include "filter_params.h"

#pragma once
#include <cuda_runtime.h>

#define CPU_FUNC __host__ __device__

struct GBuffer{
    int2 shape;
    uchar4* render = 0;
    uchar4* normal = 0;
    uchar4* albedo = 0;
    uchar4* denoised = 0;
    uchar4* buffer[2] = {0,0};

    uchar4* golden;
};

struct CpuGBuffer : GBuffer {
    Image renderImg, normalImg, albedoImg;
    CpuVector<uchar4> renderVec, albedoVec, normalVec, denoisedVec;
    CpuVector<uchar4> bufferVec;

    CpuGBuffer(){};
    CpuGBuffer (int2 shape);
    CpuGBuffer (std::string filepath);

    void allocate(int2 shape);
    void openImages(std::string filepath);

    void atrousFilterCpu(FilterParams params);
    float snr();
    void saveDenoised(std::string filepath);
};

struct CudaGBuffer : GBuffer {
    CudaVector<uchar4> renderVec, albedoVec, normalVec, denoisedVec;
    CudaVector<uchar4> bufferVec;

    CudaGBuffer(){};
    CudaGBuffer (int2 shape);

    void allocate(int2 shape);
    void openImages(std::string filepath, cudaStream_t stream = 0);
};

#endif