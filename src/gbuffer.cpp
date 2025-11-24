#include "gbuffer.h"

#include "extended_math.h"

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