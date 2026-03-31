#ifndef IMAGE_H
#define IMAGE_H

#include "utils.h"
#include "vector.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>


struct Image {
    int3 shape;
    byte* data;

    Image();

    Image(int3 shape);

    Image(byte* data, int3 shape);

    Image(std::string filename, int channels = 4);

    ~Image();

    Image& operator=(Image&& other);

    void save(std::string filename);

    static void save(std::string filename, byte* data, int3 shape);
};


#endif