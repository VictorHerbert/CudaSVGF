#pragma once
#ifndef FILTER_H
#define FILTER_H

/// @file filter.cuh
/// @brief Definition of Atrous Filters

#include "cuda_utils.h"
#include "math_utils.h"
#include "vector.h"
#include "image.h"
#include "gbuffer.h"

/// @brief Parameters controlling the cross channel filtering behavior.
struct FilterParams {
    float sigmaAlbedo = .1f;    ///< Albedo similarity weight
    float sigmaNormal = .1f;    ///< Normal similarity weight
};

#define ATROUS_RADIUS (2)                           ///< Radius of the à trous filter kernel
#define ATROUS_DIM (2 * (ATROUS_RADIUS) + 1)        ///< Kernel width
#define ATROUS_AREA ((ATROUS_DIM) * (ATROUS_DIM))   ///<  Total number of elements in the kernel

#define DEFAULT_BLOCK_SIZE dim3(32,4);              ///< Default CUDA thread block size

/// @brief Computes the separable wavelet coefficient for a given 2D offset using given spline.
/// @param pos Integer 2D offset from the kernel center.
/// @return Weight corresponding to the given offset.
INLINE CUDA_CPU_FUNC float waveletCoef(int2 pos){
    const float waveletSpline[3] = {3.0/8.0, 1.0/4.0, 1.0/16.0};
    return waveletSpline[abs(pos.x)] * waveletSpline[abs(pos.y)];
}

/// @brief Computes luminance from an RGB color.
/// @param f Input RGB color as float3.
/// @return Float  luminance value.
INLINE CUDA_CPU_FUNC 
float luminance(float3 f){
    return dot(f, make_float3(0.299, 0.587, 0.114));
}

/// @brief Selects the appropriate buffer for a given pyramid level.
/// @tparam T Buffer element type.
/// @param level Current pyramid level.
/// @param depth Maximum pyramid depth.
/// @param frame GBuffer containing render, denoised, and working buffers.
/// @return Pointer to the selected buffer for the given level.
///
/// Level mapping:
/// - level 0 → frame.render (base image)
/// - (level % 2 == depth % 2) → frame.denoised
/// - otherwise → frame.buffer
template<typename T>
INLINE CUDA_CPU_FUNC
T* getLevelBuffer(int level, int depth, GBuffer<T> frame){
    if(level == 0)
        return frame.render;

    return level%2 == depth%2 ? frame.denoised : frame.buffer;
}


/// @brief Copies a rectangular tile from an image into a tile buffer.
/// @param tile Destination tile buffer.
/// @param tileShape Dimensions of the tile (width, height).
/// @param img Source image buffer.
/// @param imgShape Dimensions of the source image (width, height).
/// @param tileStartPos Top-left coordinate of the tile in the source image.
/// @return None.
INLINE CUDA_FUNC
void saveTile(uchar4* tile, int2 tileShape, const uchar4* img, int2 imgShape, int2 tileStartPos);


/// @brief Copies a rectangular tile using alignment-optimized memory access.
/// @param tile Destination tile buffer (assumed aligned layout).
/// @param tileShape Dimensions of the tile (width, height).
/// @param img Source image buffer.
/// @param imgShape Dimensions of the source image (width, height).
/// @param tileStartPos Top-left coordinate of the tile in the source image.
/// @return None.
///
/// Optimized variant assuming row alignment or padding constraints.
INLINE CUDA_FUNC
void saveTileAligned(uchar4* tile, int2 tileShape, const uchar4* img, int2 imgShape, int2 tileStartPos);


/// @brief Execution backend for the filter pipeline.
enum FilterEngine {CPU, CUDA};

/// @brief Memory/layout strategy used in filtering.
enum FilterType {BASE, TILE, ALIGNED};


/// @brief Estimates local luminance variance using an à trous sampling pattern.
/// @tparam T Pixel type.
/// @param pos Pixel position.
/// @param in Input image buffer.
/// @param shape Image dimensions.
/// @param level Current à trous level (controls sampling radius).
/// @return Estimated local variance.
///
/// Used for edge-awareness and adaptive smoothing.
template <typename T> CUDA_CPU_FUNC
float variance(int2 pos, const T* in, int2 shape, int level);


/// @brief Executes full à trous filtering pipeline.
/// @tparam engine CPU or CUDA backend.
/// @tparam type Memory/layout strategy.
/// @tparam T Pixel type.
/// @param frame GBuffer containing all working buffers.
/// @param depth Number of pyramid levels.
/// @param params Filter configuration parameters.
/// @param stream CUDA stream (only used in CUDA backend).
/// @return None.
template <FilterEngine engine, FilterType type, typename T>
void atrousFilter(GBuffer<T> frame, int depth, FilterParams params = FilterParams(), cudaStream_t stream=0);


/// @brief CPU implementation of a single à trous filter pass.
/// @tparam type Memory/layout strategy.
/// @tparam T Pixel type.
/// @param in Input buffer.
/// @param out Output buffer.
/// @param level Current pyramid level.
/// @param frame GBuffer with auxiliary buffers.
/// @param params Filter configuration parameters.
/// @return None.
///
/// Executes one filtering iteration on CPU.
template <FilterType type, typename T>
void atrousFilterCpuPass(const T* in, T* out, int level, GBuffer<T> frame, FilterParams params = FilterParams());


/// @brief CUDA kernel for a single à trous filter pass.
/// @tparam type Memory/layout strategy.
/// @tparam T Pixel type.
/// @param in Input buffer.
/// @param out Output buffer.
/// @param level Current pyramid level.
/// @param frame GBuffer with auxiliary buffers.
/// @param params Filter configuration parameters.
/// @return None.
template <FilterType type, typename T>
KERNEL void atrousFilterCudaPass(const T* in, T* out, int level, GBuffer<T> frame, FilterParams params = FilterParams());


/// @brief Computes filtered pixel value for à trous pipeline.
/// @tparam engine CPU or CUDA backend.
/// @tparam type Memory/layout strategy.
/// @tparam T Pixel type.
/// @param pos Pixel position.
/// @param in Input buffer.
/// @param out Output buffer.
/// @param tile Optional tile buffer (used in TILE/ALIGNED modes).
/// @param level Current pyramid level.
/// @param frame GBuffer containing auxiliary buffers.
/// @param params Filter configuration parameters.
/// @return None.
template <FilterEngine engine, FilterType type, typename T>
CUDA_CPU_FUNC void atrousFilterPixel(
    int2 pos,
    const T* in,
    T* out,
    T* tile,
    int level,
    GBuffer<T> frame,
    FilterParams params = FilterParams()
);


/// @brief Instantiates CPU configuration for a specific type.
#define INSTANTIATE_ATROUS_CPU_CFG(T) \
template void atrousFilter<CPU, BASE, T>(GBuffer<T>, int, FilterParams, cudaStream_t); \
template void atrousFilterCpuPass<BASE,  T>(const T*, T*, int, GBuffer<T>, FilterParams); \
template void atrousFilterPixel<CPU, BASE, T>(int2, const T*, T*, T*, int, GBuffer<T>, FilterParams);

/// @brief Instantiates CUDA configuration for a given filter type and pixel type.
#define INSTANTIATE_ATROUS_CUDA_CFG(type, T) \
template void atrousFilter<CUDA, type, T>(GBuffer<T>, int, FilterParams, cudaStream_t); \
template KERNEL void atrousFilterCudaPass<type, T>(const T*, T*, int, GBuffer<T>, FilterParams); \
template void atrousFilterPixel<CUDA, type, T>(int2, const T*, T*, T*, int, GBuffer<T>, FilterParams);

#endif