#include "gbuffer.h"

#include "extended_math.h"

CUDA_CPU_FUNC
uchar4& Mat::operator[](int2 pos) {
    return data[flattenIndex(pos, shape)];
}

CUDA_CPU_FUNC
bool Mat::inside(int2 pos) {
    return (make_int2(0,0) <= pos) && (pos < shape);
}

CUDA_CPU_FUNC
uchar4& Tile::operator[](int2 pos) {
    return Mat::operator[](pos - start);
}

CUDA_CPU_FUNC
bool Tile::inside(int2 pos) {
    return (start <= pos) && (pos < end);
}

CUDA_CPU_FUNC
int Tile::size() {
    return totalSize(end - start);
}

CUDA_CPU_FUNC
int Tile::bytes() {
    return 4 * size();
}

CUDA_CPU_FUNC
void Tile::cache(Mat mat, int2 s, int2 e) {
    int2 dims = e - s;
    for(int y = 0; y < dims.y; y++) {
        for(int x = 0; x < dims.x; x++) {
            int2 local = make_int2(x, y);
            mat[s + local] = (*this)[local];
        }
    }
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