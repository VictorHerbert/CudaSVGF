#ifndef IMAGE_H
#define IMAGE_H

#include "cuda_utils.h"
#include "vector.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>

typedef unsigned char byte;

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