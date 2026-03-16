#ifndef VIDEO_H
#define VIDEO_H

#include "filter.cuh"

#include <string>

constexpr int MAX_STREAM = 32;

void videoFilterCpu(
    std::string filepath,
    std::string outputPath,
    int2 shape,
    int frameCount,
    FilterParams params = FilterParams()
);

void videoFilterCuda(
    std::string filepath,
    std::string outputPath,
    int2 shape,
    int frameCount,
    FilterParams params = FilterParams(),
    int maxStreamCount = 4
);


#endif