#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "primitives.h"
#include "utils.h"
#include "extended_math.h"
#include "vector.h"
#include "image.h"
#include "gbuffer.h"

#include "assert.h"

#define ATROUS_RADIUS 2

void atrousFilterPassCpu(const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params);
void atrousFilterCpu(GFrame<uchar3> frame, int depth,  FilterParams params = FilterParams());
void atrousFilterPixelCpu(int2 pos, const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params = FilterParams());

void atrousFilterCudaBase(GFrame<uchar3> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelBase(const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelBase(int2 pos, const uchar3* in, uchar3* out, int level, GFrame<uchar3> frame, FilterParams params = FilterParams());

void atrousFilterCudaU4(GFrame<uchar4> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelU4(const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelU4(int2 pos, const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());

void atrousFilterCudaO3(GFrame<uchar4> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelO3(const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelO3(int2 pos, const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());

void atrousFilterCudaOpt(GFrame<uchar4> frame, int depth, FilterParams params, cudaStream_t stream = 0);
KERNEL void atrousFilterCudaKernelOpt(const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());
CUDA_FUNC void atrousFilterPixelOpt(int2 pos, const uchar4* in, uchar4* out, int level, GFrame<uchar4> frame, FilterParams params = FilterParams());

#endif
