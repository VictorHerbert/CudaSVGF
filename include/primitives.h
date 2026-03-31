#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include <cuda_runtime.h>

struct FilterParams {
    float sigmaSpace = .5;
    float sigmaRender = 5;
    float sigmaAlbedo = 5;
    float sigmaNormal = .1;
};

template<typename T>
struct GFrame {
    int2 shape;
    T* render = nullptr;
    T* normal = nullptr;
    T* albedo = nullptr;
    T* denoised = nullptr;
    T* buffer[2] = {nullptr, nullptr};
    T* golden;
};

#endif