#ifndef VIDEO_H
#define VIDEO_H

/// @file animation.cuh
/// @brief Animation filtering

#include "filter.cuh"
#include <string>

/// @brief Applies a filter to a video using CPU processing.
/// @param filepath Input video file path.
/// @param shape Frame dimensions (width, height).
/// @param frameCount Total number of frames to process.
/// @param depth Number of iterations of the wavelet filter
/// @param params Filter configuration parameters.
void animationFilterCpu(
    std::string filepath,
    int2 shape,
    int frameCount,
    int depth,
    FilterParams params = FilterParams()
);

/// @brief Applies a filter to a video using CUDA acceleration.
/// @param filepath Input video file path.
/// @param shape Frame dimensions (width, height).
/// @param frameCount Total number of frames to process.
/// @param depth Number of iterations of the wavelet filter
/// @param params Filter configuration parameters.
/// @param maxStreamCount Maximum number of CUDA streams used for parallel processing.
void animationFilterCuda(
    std::string filepath,
    int2 shape,
    int frameCount,
    int depth,
    FilterParams params = FilterParams(),
    int maxStreamCount = 4
);


#endif