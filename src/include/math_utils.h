#ifndef EXTENDED_MATH_H
#define EXTENDED_MATH_H

#include "third_party/helper_math.h"
#include "cuda_utils.h"

/// @file math_utils.h
/// @brief Extended math utilities.

/// ---------------------------------------------------------------------------
///             Operators
/// ---------------------------------------------------------------------------

/// @brief Subtract two uchar3 vectors component-wise.
/// @param a First vector.
/// @param b Second vector.
/// @return Component-wise difference.
inline CUDA_CPU_FUNC uchar3 operator-(const uchar3 &a, const uchar3 &b) {
    return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/// @brief Subtract two uchar4 vectors component-wise.
/// @param a First vector.
/// @param b Second vector.
/// @return Component-wise difference.
inline CUDA_CPU_FUNC uchar4 operator-(const uchar4 &a, const uchar4 &b) {
    return make_uchar4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

/// @brief Scale uchar3 by scalar (scalar * vector).
/// @param f Scalar multiplier.
/// @param v Input vector.
/// @return Scaled float3 vector.
inline CUDA_CPU_FUNC float3 operator*(const float &f, const uchar3 &v) {
    return make_float3(f * v.x, f * v.y, f * v.z);
}

/// @brief Scale uchar3 by scalar (vector * scalar).
/// @param v Input vector.
/// @param f Scalar multiplier.
/// @return Scaled float3 vector.
inline CUDA_CPU_FUNC float3 operator*(const uchar3 &v, const float &f) {
    return f * v;
}

/// @brief Component-wise comparison (strictly less than).
/// @param a First 2D integer vector.
/// @param b Second 2D integer vector.
/// @return True if both components of a are less than b.
inline CUDA_CPU_FUNC bool operator<(const int2 &a, const int2 &b) {
    return (a.x < b.x) && (a.y < b.y);
}

/// @brief Component-wise less-or-equal comparison.
/// @param a First vector.
/// @param b Second vector.
/// @return True if both components of a are <= b.
inline CUDA_CPU_FUNC bool operator<=(const int2 &a, const int2 &b) {
    return (a.x <= b.x) && (a.y <= b.y);
}

/// @brief Component-wise integer division.
/// @param a Numerator vector.
/// @param b Scalar denominator.
/// @return Divided vector.
inline CUDA_CPU_FUNC int2 operator/(const int2 &a, const int &b) {
    return make_int2(a.x / b, a.y / b);
}

/// ---------------------------------------------------------------------------
///             Convertions
/// ---------------------------------------------------------------------------


/// @brief Convert arbitrary struct with x,y,z into float3.
/// @tparam T Input type with .x .y .z fields.
/// @param v Input value.
/// @return Converted float3 vector.
template<typename T>
inline CUDA_CPU_FUNC float3 make_float3(const T v) {
    return make_float3(v.x, v.y, v.z);
}

/// @brief Convert 3-component struct into uchar4 with alpha=255.
/// @tparam T Input type with .x .y .z fields.
/// @param v Input value.
/// @return uchar4 vector with alpha channel set to 255.
template<typename T>
inline CUDA_CPU_FUNC uchar4 make_uchar4(const T &v) {
    return make_uchar4(v.x, v.y, v.z, 255);
}

/// @brief Generic type conversion to uchar3 or uchar4.
/// @tparam T Target type (uchar3 or uchar4).
/// @tparam U Input type with x,y,z fields.
/// @param v Input value.
/// @return Converted vector of type T.
template<typename T, typename U>
inline CUDA_CPU_FUNC T make_type(const U &v) {
    if constexpr (std::is_same_v<T, uchar3>)
        return make_uchar3(v.x, v.y, v.z);
    else if constexpr (std::is_same_v<T, uchar4>)
        return make_uchar4(v.x, v.y, v.z, 255);
    else
        static_assert("Type must be uchar3 or uchar4");
}

/// ---------------------------------------------------------------------------
///             Utils
/// ---------------------------------------------------------------------------


/// @brief Compute total number of elements in a 2D grid.
/// @param shape Grid dimensions.
/// @return Total size (x * y).
inline CUDA_CPU_FUNC int totalSize(int2 shape) {
    return shape.x * shape.y;
}

/// @brief Check if 1D coordinate is within bounds.
/// @param pos Position index.
/// @param shape Upper bound (exclusive).
/// @return True if in range.
inline CUDA_CPU_FUNC int inRange(int pos, int shape) {
    return (pos >= 0) && (pos < shape);
}

/// @brief Check if 2D coordinate is within bounds.
/// @param pos Position vector.
/// @param shape Shape bounds.
/// @return True if both components are in range.
inline CUDA_CPU_FUNC int inRange(int2 pos, int2 shape) {
    return inRange(pos.x, shape.x) && inRange(pos.y, shape.y);
}

/// @brief Convert 2D coordinates to flat index.
/// @param p 2D position.
/// @param shape Grid shape.
/// @return Flattened index.
inline CUDA_CPU_FUNC int posToIndex(int2 p, int2 shape) {
    return p.y * shape.x + p.x;
}

/// @brief Convert flat index to 2D coordinates.
/// @param p Flat index.
/// @param shape Grid shape.
/// @return 2D coordinates.
inline CUDA_CPU_FUNC int2 indexToPos(int p, int2 shape) {
    return {p % shape.x, p / shape.x};
}

/// @brief Convert encoded normal vector to normalized float3.
/// @tparam T Input vector type with x,y,z fields.
/// @param v Encoded normal.
/// @return Normalized float3 in [-1, 1] range.
template <typename T>
inline CUDA_CPU_FUNC float3 parseNormal(T v) {
    return normalize(make_float3(v) - make_float3(255, 255, 255) / 2);
}

#endif