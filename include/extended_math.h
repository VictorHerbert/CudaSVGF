#ifndef EXTENDED_MATH_H
#define EXTENDED_MATH_H

#include "third_party/helper_math.h"

inline CUDA_CPU_FUNC uchar3 operator-(const uchar3 &a, const uchar3 &b) {
    return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline CUDA_CPU_FUNC float length(const uchar3 &v) {
    return sqrtf(float(v.x) * float(v.x) +
                 float(v.y) * float(v.y) +
                 float(v.z) * float(v.z));
}

inline CUDA_CPU_FUNC float dot(const uchar3 &a, const uchar3 &b) {
    return float(a.x) * float(b.x) +
           float(a.y) * float(b.y) +
           float(a.z) * float(b.z);
}

inline CUDA_CPU_FUNC float3 operator*(const float &f, const uchar3 &v) {
    return make_float3(f * v.x, f * v.y, f * v.z);
}

inline CUDA_CPU_FUNC float3 operator*(const uchar3 &v, const float &f) {
    return f * v;
}

template<typename T>
CUDA_CPU_FUNC float3 make_float3(const T v){
    return make_float3(v.x, v.y, v.z);
}

template<typename T>
inline CUDA_CPU_FUNC uchar4 make_uchar4(const T &v){
    /*return make_uchar4(
        static_cast<unsigned char>(v.x),
        static_cast<unsigned char>(v.y),
        static_cast<unsigned char>(v.y),
        0
    );*/

    return make_uchar4(v.x, v.y, v.z, 0);
}

/*
template<typename T>
inline CUDA_CPU_FUNC uchar3 make_uchar3(const T &v) {
    return make_uchar3(v.x, v.y, v.z);
}

inline CUDA_CPU_FUNC uchar4 make_uchar4( uchar3 &v) {
    return make_uchar4(v.x, v.y, v.z, 0);
}

inline CUDA_CPU_FUNC uchar4 make_uchar4(const float3 &v) {
    return make_uchar4(v.x, v.y, v.z, 0);
}*/



inline CUDA_CPU_FUNC float length2(const float3 &v){
    return length(v)*length(v);
}

inline CUDA_CPU_FUNC int totalSize(int2 shape){
    return shape.x * shape.y;
}

inline CUDA_CPU_FUNC int totalSize(int3 shape){
    return shape.x * shape.y * shape.z;
}

inline CUDA_CPU_FUNC int inRange(int2 pos, int2 shape){
    return (pos.x >= 0) && (pos.x < shape.x) && (pos.y >= 0) && (pos.y < shape.y);
}

inline CUDA_CPU_FUNC int flattenIndex(int2 p, int2 shape){
    return p.y * shape.x + p.x;
}

#endif