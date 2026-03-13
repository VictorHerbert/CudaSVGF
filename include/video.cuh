#ifndef VIDEO_H
#define VIDEO_H

#include "filter.cuh"

#include <string>

void videoFilterCpu(std::string filepath, const FilterParams params);


#endif