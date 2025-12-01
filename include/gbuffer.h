#ifndef GBUFFER_H
#define GBUFFER_H

#include "image.h"
#include "extended_math.h"

#pragma once
#include <cuda_runtime.h>

#define CPU_FUNC __host__ __device__

/**
 * @brief Computes the flattened linear index for a 2D position in a matrix.
 * @param pos 2D coordinate
 * @param shape Matrix size {width, height}
 * @return Linear index into the data array
 */
CPU_FUNC int flattenIndex(int2 pos, int2 shape);

/**
 * @brief Computes the total number of elements in a 2D region.
 * @param dims Dimensions {width, height}
 * @return Number of elements
 */
CPU_FUNC int totalSize(int2 dims);

/**
 * @brief Matrix wrapper storing a 2D array of uchar4 pixels.
 *
 * Provides bounds checking and 2D indexing.
 */
struct Mat {
    int2 shape;
    uchar4* data;

    /**
     * @brief Returns reference to pixel at given coordinates.
     */
    CPU_FUNC uchar4& operator[](int2 pos);

    /**
     * @brief Checks whether a position falls inside the matrix dimensions.
     */
    CPU_FUNC bool inside(int2 pos);
};

/**
 * @brief A subregion (tile) of a parent matrix.
 *
 * Allows addressing a subset of Mat, with automatic coordinate translation.
 */
struct Tile : Mat {
    uchar4* tile;
    int2 start;
    int2 end;

    /**
     * @brief Saves the tile data into a region of another matrix.
     * @param mat Parent matrix
     * @param s Start coordinate
     * @param e End coordinate
     */
    CPU_FUNC void cache(Mat mat, int2 s, int2 e);

    /**
     * @brief Returns reference to pixel using tile coordinates.
     */
    CPU_FUNC uchar4& operator[](int2 pos);

    /**
     * @brief Checks whether a global coordinate lies inside the tile.
     */
    CPU_FUNC bool inside(int2 pos);

    /**
     * @brief Number of elements in the tile.
     */
    CPU_FUNC int size();

    /**
     * @brief Total byte size of the tile (uchar4).
     */
    CPU_FUNC int bytes();
};

struct GBuffer{
    int2 shape;
    uchar4* render;
    uchar4* normal;
    uchar4* albedo;
    uchar4* denoised;
    uchar4* buffer[2];
    
    uchar4* renderTile;
    uchar4* normalTile;
    uchar4* albedoTile;
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