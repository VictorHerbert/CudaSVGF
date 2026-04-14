#ifndef GBUFFER_H
#define GBUFFER_H

/// @file gbuffer.h
/// @brief CPU and CUDA G-buffer frame abstractions

#include "image.h"
#include "math_utils.h"

#include <regex>
#include <cuda_runtime.h>

/// @brief Generic G-buffer storing multiple render targets
/// @tparam T Element type stored in buffers
template<typename T>
struct GBuffer {
    int2 shape = {0,0};

    T* render = nullptr;
    T* normal = nullptr;
    T* albedo = nullptr;
    T* denoised = nullptr;

    T* buffer = nullptr;
};

/// @brief CPU-side G-buffer implementation
/// @tparam T Element type stored in buffers
template<typename T>
struct CpuGBuffer : GBuffer<T> {
    CpuVector<T> renderVec;
    CpuVector<T> albedoVec;
    CpuVector<T> normalVec;
    CpuVector<T> denoisedVec;
    CpuVector<T> bufferVec;

    CpuGBuffer() = default;

    /// @brief Constructs a CPU G-buffer with given resolution
    /// @param shape Frame resolution (width, height)
    explicit CpuGBuffer(int2 shape);

    /// @brief Resizes CPU buffers
    /// @param shape New frame resolution (width, height)
    void resize(int2 shape);
};

template<typename T>
CpuGBuffer<T>::CpuGBuffer(int2 shape) {
    this->resize(shape);
}

template<typename T>
void CpuGBuffer<T>::resize(int2 shape) {
    int size = totalSize(shape);
    this->shape = shape;

    renderVec.resize(size);
    albedoVec.resize(size);
    normalVec.resize(size);
    denoisedVec.resize(size);
    bufferVec.resize(size);

    this->render = renderVec.data();
    this->denoised = denoisedVec.data();
    this->albedo = albedoVec.data();
    this->normal = normalVec.data();
    this->buffer = bufferVec.data();
}

/// @brief CUDA-side G-buffer implementation
/// @tparam T Element type stored in buffers
template<typename T>
struct CudaGBuffer : GBuffer<T> {
    CudaVector<T> renderVec;
    CudaVector<T> albedoVec;
    CudaVector<T> normalVec;
    CudaVector<T> denoisedVec;
    CudaVector<T> bufferVec;

    CudaGBuffer() = default;

    /// @brief Constructs a CUDA G-buffer with given resolution
    /// @param shape Frame resolution (width, height)
    explicit CudaGBuffer(int2 shape);

    /// @brief Resizes CUDA buffers
    /// @param shape New frame resolution (width, height)
    void resize(int2 shape);
};

template<typename T>
CudaGBuffer<T>::CudaGBuffer(int2 shape) {
    this->resize(shape);
}

template<typename T>
void CudaGBuffer<T>::resize(int2 shape) {
    int size = totalSize(shape);
    this->shape = shape;

    renderVec.resize(size);
    albedoVec.resize(size);
    normalVec.resize(size);
    denoisedVec.resize(size);
    bufferVec.resize(size);

    this->render = renderVec.data();
    this->denoised = denoisedVec.data();
    this->albedo = albedoVec.data();
    this->normal = normalVec.data();
    this->buffer = bufferVec.data();
}

#endif