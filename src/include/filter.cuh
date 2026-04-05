#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "cuda_utils.h"
#include "math_utils.h"
#include "vector.h"
#include "image.h"
#include "gframe.h"

#include "assert.h"

struct FilterParams {
    float sigmaSpace = .5;
    float sigmaRender = 5;
    float sigmaAlbedo = 5;
    float sigmaNormal = .1;
};

#define ATROUS_RADIUS (2)
#define ATROUS_DIM (2 * (ATROUS_RADIUS) + 1)
#define ATROUS_AREA ((ATROUS_DIM) * (ATROUS_DIM))

void atrousFilterPassCpu(const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params);
void atrousFilterCpu(GFrame<uchar3> frame, int depth,  FilterParams params = FilterParams());
void atrousFilterPixelCpu(int2 pos, const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params = FilterParams());

void atrousFilterCudaNoOpt(GFrame<uchar3> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelNoOpt(const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelNoOpt(int2 pos, const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params = FilterParams());

void atrousFilterCudaBase(GFrame<uchar3> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelBase(const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelBase(int2 pos, const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params = FilterParams());

void atrousFilterCudaU4(GFrame<uchar4> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelU4(const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelU4(int2 pos, const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());

void atrousFilterCudaAprox(GFrame<uchar4> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelAprox(const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelAprox(int2 pos, const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());

void atrousFilterCudaTile(GFrame<uchar4> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelTile(const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelTile(int2 pos, uchar4* tile, const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());

void atrousFilterCudaTileAligned(GFrame<uchar4> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelTileAligned(const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelTileAligned(int2 pos, uchar4* tile, const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());


#endif
