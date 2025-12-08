#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "utils.h"
#include "extended_math.h"
#include "vector.h"
#include "image.h"
#include "gbuffer.h"

struct FilterParams {
    int depth = 1;
    int radius = 2;

    float sigmaSpace = .1;
    float sigmaColor = .1;
    float sigmaAlbedo = .1;
    float sigmaNormal = .1;
};

CUDA_CPU_FUNC int haloSize(int radius, int level);

CUDA_CPU_FUNC int fullTileSize(int radius, int2 blockDimX, int level);
CUDA_CPU_FUNC int fullTileArea(int radius, int level);
CUDA_CPU_FUNC int fullTileBytes(int radius, int level);

CUDA_CPU_FUNC int lineTileSize(int radius, int blockDimX, int level);
CUDA_CPU_FUNC int lineTileArea(int radius, int2 blockDim, int level);
CUDA_CPU_FUNC int lineTileBytes(int radius, int2 blockDim, int level);

CUDA_FUNC int tileLine(uchar4* tile, uchar4* input, int size);

CUDA_FUNC int tileLine(uchar4* tile, uchar4* input, int2 shape, int radius, int line, int level);

CUDA_FUNC   void dilatedFilterBase      (uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);
CUDA_FUNC   void dilatedFilter2      (uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);

//CUDA_FUNC   void dilatedFilterFullTile  (uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);
CUDA_FUNC   void dilatedFilterLineTile  (uchar4* in, uchar4* out, int level, const GBuffer frame, const FilterParams params);

#endif