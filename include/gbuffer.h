#ifndef GBUFFER_H
#define GBUFFER_H

#include "image.h"

struct GBuffer{
    int2 shape;

    uchar4* render;
    uchar4* denoised;
    uchar4* normal;
    uchar4* albedo;
    uchar4* buffer[2];
};

struct CPUGBuffer : GBuffer {
    Image render, albedo, normal;
};

struct CudaGBuffer : GBuffer {
    CudaVector<uchar4> renderVec, albedoVec, normalVec, denoisedVec;
    CudaVector<uchar4> bufferVec;
    //CpuVector<Pixel> denoisedVecCpu; // TODO remove
    uchar4* denoisedCPU;

    CudaGBuffer(){};
    //~CudaGBuffer();
    CudaGBuffer (int2 shape);

    void allocate(int2 shape);

    void openImages(std::string filepath, cudaStream_t stream = 0);
};

#endif