#ifndef GBUFFER_H
#define GBUFFER_H

#include "primitives.h"
#include "image.h"
#include "extended_math.h"

#include <regex>
#include <cuda_runtime.h>


template<typename T>
struct CpuGBuffer : GBuffer<T> {
    Image renderImg, normalImg, albedoImg;
    CpuVector<T> renderVec, albedoVec, normalVec, denoisedVec;
    CpuVector<T> bufferVec;

    CpuGBuffer(){};
    CpuGBuffer (int2 shape);
    CpuGBuffer (std::string filepath);

    void resize(int2 shape);
    void openImages(std::string filepath);

    void atrousFilterCpu(FilterParams params);
    float snr();
    void saveDenoised(std::string filepath);
};

template<typename T>
struct CudaGBuffer : GBuffer<T> {
    CudaVector<T> renderVec, albedoVec, normalVec, denoisedVec;
    CudaVector<T> bufferVec;

    CudaGBuffer(){};
    CudaGBuffer (int2 shape);

    void resize(int2 shape);
    void openImages(std::string filepath, cudaStream_t stream = 0);
};

template<typename T>
CpuGBuffer<T>::CpuGBuffer (int2 shape){
    this->resize(shape);
}

template<typename T>
CpuGBuffer<T>::CpuGBuffer(std::string filepath){
    openImages(filepath);
    this->resize({renderImg.shape.x, renderImg.shape.y});
}

template<typename T>
void CpuGBuffer<T>::openImages(std::string filepath){
    renderImg = Image(std::regex_replace(filepath, std::regex("\\$channel\\$"), "render"), 4);
    normalImg = Image(std::regex_replace(filepath, std::regex("\\$channel\\$"), "normal"), 4);
    albedoImg = Image(std::regex_replace(filepath, std::regex("\\$channel\\$"), "albedo"), 4);

    this->render = (T*) renderImg.data;
    this->normal = (T*) normalImg.data;
    this->albedo = (T*) albedoImg.data;
}

template<typename T>
void CpuGBuffer<T>::resize(int2 shape){
    int size = totalSize(shape);
    this->shape = shape;
    denoisedVec.resize(size);
    bufferVec.resize(2*size);
    
    this->denoised = denoisedVec.data();
    this->buffer[0] = bufferVec.data();
    this->buffer[1] = bufferVec.data() + size;
}

template<typename T>
void CpuGBuffer<T>::saveDenoised(std::string filepath){
    Image::save(filepath, (byte*) this->denoised, {this->shape.x, this->shape.y, 4});
}

template<typename T>
CudaGBuffer<T>::CudaGBuffer (int2 shape){
    this->resize(shape);
}

template<typename T>
void CudaGBuffer<T>::resize(int2 shape){
    int size = totalSize(shape);
    this->shape = shape;
    renderVec.resize(size);
    albedoVec.resize(size);
    normalVec.resize(size);
    denoisedVec.resize(size);
    bufferVec.resize(2*size); // TODO make two allocations to guarantee alignment

    this->render = renderVec.data();
    this->denoised = denoisedVec.data();
    this->albedo = albedoVec.data();
    this->normal = normalVec.data();
    this->buffer[0] = bufferVec.data();
    this->buffer[1] = bufferVec.data() + size;
}

#endif