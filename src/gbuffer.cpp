#include "gbuffer.h"

#include "extended_math.h"
#include "filter.cuh"

#include <regex>

CpuGBuffer::CpuGBuffer (int2 shape){
    this->allocate(shape);
}

CpuGBuffer::CpuGBuffer(std::string filepath){
    openImages(filepath);
    this->allocate({renderImg.shape.x, renderImg.shape.y});
}

void CpuGBuffer::openImages(std::string filepath){
    renderImg = Image(std::regex_replace(filepath, std::regex("\\$channel\\$"), "render"), 4);
    normalImg = Image(std::regex_replace(filepath, std::regex("\\$channel\\$"), "normal"), 4);
    albedoImg = Image(std::regex_replace(filepath, std::regex("\\$channel\\$"), "albedo"), 4);

    render = (uchar4*) renderImg.data;
    normal = (uchar4*) normalImg.data;
    albedo = (uchar4*) albedoImg.data;
}

void CpuGBuffer::allocate(int2 shape){
    int size = totalSize(shape);
    this->shape = shape;
    denoisedVec.resize(size);
    bufferVec.resize(2*size);
    
    denoised = denoisedVec.data();
    buffer[0] = bufferVec.data();
    buffer[1] = bufferVec.data() + size;
}

//void CpuGBuffer::atrousFilterCpu(FilterParams params){
//    atrousFilterCpu((GBuffer) *this, params);
//}

float CpuGBuffer::snr(){
    return 1;
}

void CpuGBuffer::saveDenoised(std::string filepath){
    Image::save(filepath, (byte*) denoised, {shape.x, shape.y, 4});
}

CudaGBuffer::CudaGBuffer (int2 shape){
    this->allocate(shape);
}

void CudaGBuffer::allocate(int2 shape){
    int size = totalSize(shape);
    this->shape = shape;
    renderVec.resize(size);
    albedoVec.resize(size);
    normalVec.resize(size);
    denoisedVec.resize(size);
    bufferVec.resize(2*size);
    
    render = renderVec.data();
    denoised = denoisedVec.data();
    albedo = albedoVec.data();
    normal = normalVec.data();
    buffer[0] = bufferVec.data();
    buffer[1] = bufferVec.data() + size;

}