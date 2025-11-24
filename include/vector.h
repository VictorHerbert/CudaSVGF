#ifndef VECTOR_H
#define VECTOR_H

#include <string>
#include <vector>
#include <stdio.h>

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename T>
using CpuVector = std::vector<T>;

/**
 * @brief Alias for a standard CPU-side dynamic array.
 * @tparam T Element type.
 */
template <typename T>
using CpuVector = std::vector<T>;

/**
 * @brief GPU-managed dynamic array backed by CUDA device memory.
 * @tparam T Element type.
 *
 * Provides allocation, deallocation, and data transfer between host
 * (CpuVector) and device memory. Instances manage their own lifetime
 * and ensure proper cleanup of GPU resources. The interface mirrors
 * familiar parts of std::vector by exposing commonly used methods such
 * as size and data for easier integration with existing code.
 */

template <typename T>
struct CudaVector {
private:
    T* data_p;          /**< Pointer to device memory. */
    size_t size_p;      /**< Number of elements stored. */

public:
    /**
     * @brief Constructs an empty GPU vector with no allocated memory.
     */
    CudaVector();

    /**
     * @brief Allocates GPU memory for a given number of elements.
     * @param size Number of elements to allocate.
     */
    CudaVector(size_t size);

    /**
     * @brief Wraps an existing device pointer without allocating memory.
     * @param v Existing device pointer.
     * @param size Number of elements available at the pointer.
     */
    CudaVector(T* v, size_t size);

    /**
     * @brief Allocates and initializes device memory from a CPU vector.
     * @param v Host-side vector providing initial contents.
     */
    CudaVector(CpuVector<T>& v);

    /**
     * @brief Returns the raw device pointer.
     */
    T* data();

    /**
     * @brief Returns the raw device pointer as a const pointer.
     */
    const T* data() const;

    /**
     * @brief Returns the number of elements stored.
     */
    size_t size() const;

    /**
     * Resizes the device buffer. Existing contents are preserved up to
     * the minimum of the old and new sizes.
     * @param size New size
     */
    void resize(size_t size);


    /**
     * @brief Copies data from a host pointer into the device buffer.
     * @param v Host pointer containing the source data.
     * @param size Number of elements to copy.
     */
    void copyFrom(T* v, size_t size);

    /**
     * @brief Asynchronously copies data from a host pointer into the device buffer.
     * @param v Host pointer containing the source data.
     * @param size Number of elements to copy.
     * @param stream CUDA stream on which to perform the transfer.
     */
    void copyFromAsync(T* v, size_t size, cudaStream_t stream);

    /**
     * @brief Copies data from the device buffer into a host pointer.
     * @param v Host pointer to store the copied elements.
     */
    void copyTo(T* v);

    /**
     * @brief Asynchronously copies data from the device buffer into a host pointer.
     * @param v Host pointer to store the copied elements.
     * @param stream CUDA stream on which to perform the transfer.
     */
    void copyToAsync(T* v, cudaStream_t stream);


    /**
     * @brief Releases all associated GPU memory.
     */
    ~CudaVector();
};


template <typename T>
CudaVector<T>::CudaVector() : size_p(0), data_p(nullptr) {}

template <typename T>
CudaVector<T>::CudaVector(size_t size) : size_p(size) {
    cudaMalloc(&data_p, size_p * sizeof(T));
}

template <typename T>
CudaVector<T>::CudaVector(T* v, size_t size) : size_p(size) {
    cudaMalloc(&data_p, size_p * sizeof(T));
    cudaMemcpy(data_p, v, size_p * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
CudaVector<T>::CudaVector(CpuVector<T>& v) : CudaVector(v.data(), v.size()) {}

template <typename T>
T* CudaVector<T>::data() {return data_p;}

template <typename T>
const T* CudaVector<T>::data() const {return data_p;}

template <typename T>
size_t CudaVector<T>::size() const {return size_p;}

template <typename T>
void CudaVector<T>::resize(size_t newSize) {
    if (data_p)
        cudaFree(data_p);
    cudaMalloc(&data_p, newSize * sizeof(T));
    size_p = newSize;
}

template <typename T>
void CudaVector<T>::copyFrom(T* v, size_t size) {
    if (size > size_p)
        throw std::runtime_error("Size mismatch");
    cudaMemcpy(data_p, v, size * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void CudaVector<T>::copyFromAsync(T* v, size_t size, cudaStream_t stream) {
    if (size > size_p)
        throw std::runtime_error("Size mismatch");
    cudaMemcpyAsync(data_p, v, size * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template <typename T>
void CudaVector<T>::copyTo(T *v) {
    cudaMemcpy(v, data_p, sizeof(T) * size_p, cudaMemcpyDeviceToHost);
}

template <typename T>
void CudaVector<T>::copyToAsync(T *v, cudaStream_t stream) {
    cudaMemcpyAsync(v, data_p, sizeof(T) * size_p, cudaMemcpyDeviceToHost, stream);
}

template <typename T>
CudaVector<T>::~CudaVector() {
    if(size_p != 0)
        cudaFree(data_p);
}



#endif