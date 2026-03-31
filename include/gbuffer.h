#ifndef GBUFFER_H
#define GBUFFER_H

#include "primitives.h"
#include "image.h"
#include "extended_math.h"

#include <regex>
#include <cuda_runtime.h>


template<typename T>
struct CpuGFrame : GFrame<T> {
    Image renderImg, normalImg, albedoImg;
    CpuVector<T> renderVec, albedoVec, normalVec, denoisedVec;
    CpuVector<T> bufferVec;

    CpuGFrame(){};
    CpuGFrame (int2 shape);
    CpuGFrame (std::string filepath);

    void resize(int2 shape);
};

template<typename T>
struct CudaGFrame : GFrame<T> {
    CudaVector<T> renderVec, albedoVec, normalVec, denoisedVec;
    CudaVector<T> bufferVec;

    CudaGFrame(){};
    CudaGFrame (int2 shape);

    void resize(int2 shape);
};

template<typename T>
CpuGFrame<T>::CpuGFrame (int2 shape){
    this->resize(shape);
}

template<typename T>
void CpuGFrame<T>::resize(int2 shape){
    int size = totalSize(shape);
    this->shape = shape;
    denoisedVec.resize(size);
    bufferVec.resize(2*size);
    
    this->denoised = denoisedVec.data();
    this->buffer[0] = bufferVec.data();
    this->buffer[1] = bufferVec.data() + size;
}

template<typename T>
CudaGFrame<T>::CudaGFrame (int2 shape){
    this->resize(shape);
}

template<typename T>
void CudaGFrame<T>::resize(int2 shape){
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