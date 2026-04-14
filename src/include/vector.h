#ifndef VECTOR_H
#define VECTOR_H

/// @file vector.h
/// @brief CPU and CUDA vector abstractions for host-device memory management.

#include <string>
#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>

/// @brief Alias for CPU-side vector storage.
template <typename T>
using CpuVector = std::vector<T>;

/// @brief CUDA-managed dynamic array.
/// @tparam T Element type stored in device memory.
template <typename T>
struct CudaVector {
private:
    T* data_p;
    size_t size_p;

public:
    /// @brief Default empty constructor.
    CudaVector();

    /// @brief Allocates device memory for a given number of elements.
    CudaVector(size_t size);

    /// @brief Device pointer accessor.
    /// @return Raw pointer to device memory.
    T* data();

    /// @brief Reallocates device memory.
    /// Frees existing memory if allocated.
    /// @param size New size in elements.
    void resize(size_t size);

    /// @brief Copies data from host to device (asynchronous).
    /// @param v Host source pointer.
    /// @param size Number of elements to copy.
    /// @param stream CUDA stream for async execution.
    void copyFromAsync(T* v, size_t size, cudaStream_t stream=0);

    /// @brief Copies data from device to host (asynchronous).
    /// @param v Host destination pointer.
    /// @param stream CUDA stream for async execution.
    void copyToAsync(T* v, cudaStream_t stream=0);

    /// @brief Destructor.
    /// Frees device memory if allocated.
    ~CudaVector();
};

// ===================== Implementation =====================

template <typename T>
CudaVector<T>::CudaVector() : size_p(0), data_p(nullptr) {}

template <typename T>
CudaVector<T>::CudaVector(size_t size) : size_p(size) {
    cudaMalloc(&data_p, size_p * sizeof(T));
}

template <typename T>
T* CudaVector<T>::data() { return data_p; }

template <typename T>
void CudaVector<T>::resize(size_t newSize) {
    if (data_p) {
        cudaFree(data_p);
    }

    CUDA_ERROR_CHECK(cudaMalloc(&data_p, newSize * sizeof(T)));
    size_p = newSize;
}


template <typename T>
void CudaVector<T>::copyFromAsync(T* v, size_t size, cudaStream_t stream) {
    if (size > size_p)
        throw std::runtime_error("Size mismatch");

    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        data_p,
        v,
        size * sizeof(T),
        cudaMemcpyHostToDevice,
        stream
    ));
}

template <typename T>
void CudaVector<T>::copyToAsync(T* v, cudaStream_t stream) {
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        v,
        data_p,
        sizeof(T) * size_p,
        cudaMemcpyDeviceToHost,
        stream
    ));
}

template <typename T>
CudaVector<T>::~CudaVector() {
    if (data_p) {
        cudaFree(data_p);
    }
}

#endif