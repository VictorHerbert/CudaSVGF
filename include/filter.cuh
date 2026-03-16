#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "utils.h"
#include "extended_math.h"
#include "vector.h"
#include "image.h"
#include "gbuffer.h"
#include "filter_params.h"

CUDA_CPU_FUNC void atrousFilterPixel(int2 pos, uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);

void atrousFilterCpu(GBuffer frame, const FilterParams params);
void atrousFilterPassCpu(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);

KERNEL void atrousFilterCudaBase(GBuffer frame, const FilterParams params);
CUDA_FUNC void atrousFilterPassCudaBase(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);

//CUDA_FUNC void AtrousFilterCuda(GBuffer frame, const FilterParams params);
//CUDA_FUNC void AtrousFilterPassCuda(uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);

#endif
