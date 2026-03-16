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
    uchar4* render = nullptr;
    uchar4* normal = nullptr;
    uchar4* albedo = nullptr;
    uchar4* denoised = nullptr;
    uchar4* buffer[2] = {nullptr, nullptr};

    uchar4* golden;
};

struct CpuGBuffer : GBuffer {
    Image renderImg, normalImg, albedoImg;
    CpuVector<uchar4> renderVec, albedoVec, normalVec, denoisedVec;
    CpuVector<uchar4> bufferVec;

    CpuGBuffer(){};
    CpuGBuffer (int2 shape);
    CpuGBuffer (std::string filepath);

    void resize(int2 shape);
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

    void resize(int2 shape);
    void openImages(std::string filepath, cudaStream_t stream = 0);
};

#endif