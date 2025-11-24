#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "utils.h"
#include "extended_math.h"
#include "vector.h"
#include "image.h"
#include "gbuffer.h"

struct FilterParams {
    //enum FilterType {AVERAGE, GAUSSIAN, CROSS, WAVELET} type;
    enum {
        AVERAGE     = 0,
        SPATIAL     = 1 << 0,
        RENDER      = 1 << 1,
        ALBEDO      = 1 << 2,
        NORMAL      = 1 << 3,
        WAVELET     = 1 << 4,
        BILATERAL   = SPATIAL | RENDER,
        CROSS       = SPATIAL | RENDER | ALBEDO | NORMAL
    } type = CROSS;

    int depth;
    int radius = 2;

    float sigmaSpace = .1;
    float sigmaColor = .1;
    float sigmaAlbedo = .1;
    float sigmaNormal = .1;

    bool cacheTile = false;
};


CUDA_CPU_FUNC int tileByteCount(FilterParams params, uint2 blockShape);

KERNEL      void filterKernel       (GBuffer frame, FilterParams params);
CUDA_FUNC   void singleLevelFilter  (uchar4* in, uchar4* out, uchar4* tile, const GBuffer frame, const FilterParams params);
CUDA_FUNC   void cacheTile          (uchar4* tile, const uchar4* in, const int2 frameShape, const int2 start, const int2 end);

#endif