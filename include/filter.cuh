#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "utils.h"
#include "extended_math.h"
#include "vector.h"
#include "image.h"
#include "gbuffer.h"

struct FilterParams {

    enum Type {
        AVERAGE     = 0,
        SPATIAL     = 1 << 0,
        RENDER      = 1 << 1,
        ALBEDO      = 1 << 2,
        NORMAL      = 1 << 3,
        WAVELET     = 1 << 4,
        BILATERAL   = SPATIAL | RENDER,
        CROSS       = SPATIAL | RENDER | ALBEDO | NORMAL
    };

    Type type = CROSS;
    Type tile = static_cast<Type>(ALBEDO | NORMAL);
    int2 tileShape;

    int depth = 1;
    int radius = 2;

    float sigmaSpace = .1;
    float sigmaColor = .1;
    float sigmaAlbedo = .1;
    float sigmaNormal = .1;
};

CUDA_CPU_FUNC int haloSize(int radius, int level);
//CUDA_CPU_FUNC int tileByteCount(int radius, int level, int2 blockShape);
CUDA_CPU_FUNC int tileBytes(FilterParams params, int2 blockShape);

CUDA_FUNC   void cacheTile              (uchar4* tile, const uchar4* in, const int2 frameShape, const int2 start, const int2 end);

KERNEL      void filterKernelBase           (GBuffer frame, FilterParams params);
KERNEL      void filterKernel           (GBuffer frame, FilterParams params);
CUDA_FUNC   void singleLevelFilter      (uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);



CUDA_FUNC   void singleLevelFilterBase  (uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);

#endif