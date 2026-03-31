#ifndef EXTENDED_MATH_H
#define EXTENDED_MATH_H

#include "third_party/helper_math.h"

#include <type_traits>


// Operators

inline CUDA_CPU_FUNC uchar3 operator-(const uchar3 &a, const uchar3 &b) {
    return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline CUDA_CPU_FUNC uchar4 operator-(const uchar4 &a, const uchar4 &b) {
    return make_uchar4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline CUDA_CPU_FUNC float3 operator * (const float &f, const uchar3 &v) {
    return make_float3(f * v.x, f * v.y, f * v.z);
}

inline CUDA_CPU_FUNC float3 operator * (const uchar3 &v, const float &f) {
    return f * v;
}

inline CUDA_CPU_FUNC bool operator < (const int2 &a, const int2 &b) {
    return (a.x < b.x) && (a.y < b.y);
}

inline CUDA_CPU_FUNC bool operator <= (const int2 &a, const int2 &b) {
    return (a.x <= b.x) && (a.y <= b.y);
}

inline CUDA_CPU_FUNC int2 operator / (const int2 &a, const int &b) {
    return make_int2(a.x/b, a.y/b);
}


// Convertions

template<typename T>
inline CUDA_CPU_FUNC float3 make_float3(const T v){
    return make_float3(v.x, v.y, v.z);
}

template<typename T>
inline CUDA_CPU_FUNC uchar4 make_uchar4(const T &v){
    return make_uchar4(v.x, v.y, v.z, 255);
}

template<typename T, typename U>
inline CUDA_CPU_FUNC T make_type(const U &v){
    if constexpr (std::is_same_v<T, uchar3>)
        return make_uchar3(v.x, v.y, v.z);
    else if constexpr (std::is_same_v<T, uchar4>)
        return make_uchar4(v.x, v.y, v.z, 255);
    else
        static_assert("Type must be uchar3 or uchar4");
}

// Utils

inline CUDA_CPU_FUNC float length2(const uchar4 &v){
    return 
        float(v.x) * float(v.x) +
        float(v.y) * float(v.y) +
        float(v.z) * float(v.z);
}

inline CUDA_CPU_FUNC float length2(const float3 &v){
    return 
        float(v.x) * float(v.x) +
        float(v.y) * float(v.y) +
        float(v.z) * float(v.z);
}



inline CUDA_CPU_FUNC int totalSize(int2 shape){
    return shape.x * shape.y;
}

inline CUDA_CPU_FUNC int totalSize(int3 shape){
    return shape.x * shape.y * shape.z;
}

inline CUDA_CPU_FUNC int inRange(int pos, int shape){
    return (pos >= 0) && (pos < shape);
}

inline CUDA_CPU_FUNC int inRange(int2 pos, int2 shape){
    return inRange(pos.x, shape.x) && inRange(pos.y, shape.y);
}

inline CUDA_CPU_FUNC int flattenIndex(int2 p, int2 shape){
    return p.y * shape.x + p.x;
}

inline CUDA_CPU_FUNC int2 indexToPos(int p, int2 shape){
    return {p%shape.x, p/shape.x};
}

#endif