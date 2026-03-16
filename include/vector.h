#ifndef VECTOR_H
#define VECTOR_H

#include <string>
#include <vector>
//#include <stdio.h>

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename T>
using CpuVector = std::vector<T>;

template <typename T>
struct CudaVector {
private:
    T* data_p;
    size_t size_p;

public:
    CudaVector();

    CudaVector(size_t size);

    CudaVector(T* v, size_t size);

    CudaVector(CpuVector<T>& v);

    T* data();

    const T* data() const;

    size_t size() const;

    void resize(size_t size);

    void copyFrom(T* v, size_t size);

    void copyFromAsync(T* v, size_t size, cudaStream_t stream);

    void copyTo(T* v);

    void copyToAsync(T* v, cudaStream_t stream);

    ~CudaVector();
};


template <typename T>
CudaVector<T>::CudaVector() : size_p(0), data_p(nullptr) {}

template <typename T>
CudaVector<T>::CudaVector(size_t size) : size_p(size) {
    cudaMalloc(&data_p, size_p * sizeof(T));
    //printf("cudaMalloc at %p\n", data_p);
}

template <typename T>
CudaVector<T>::CudaVector(T* v, size_t size) : size_p(size) {
    cudaMalloc(&data_p, size_p * sizeof(T));
    cudaMemcpy(data_p, v, size_p * sizeof(T), cudaMemcpyHostToDevice);
    //printf("cudaMalloc at %p\n", data_p);
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
    if (data_p){
        cudaFree(data_p);
        //printf("cudaFree at %p\n", data_p);
    }
    CUDA_ERROR_CHECK(
        cudaMalloc(&data_p, newSize * sizeof(T)));
    //printf("cudaMalloc at %p\n", data_p);
    size_p = newSize;
}

template <typename T>
void CudaVector<T>::copyFrom(T* v, size_t size) {
    if (size > size_p)
        throw std::runtime_error("Size mismatch");
    CUDA_ERROR_CHECK(
        cudaMemcpy(data_p, v, size * sizeof(T), cudaMemcpyHostToDevice));
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
    if(data_p){
        cudaFree(data_p);
        //printf("cudaFree at %p\n", data_p);
    }
}



#endif