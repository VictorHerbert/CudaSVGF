#include "image.h"
#include "cuda_utils.h"
#include "math_utils.h"


void* cuda_malloc(size_t size) {
    void* ptr = NULL;
    if (cudaMallocHost(&ptr, size) != cudaSuccess)
        return NULL;
    return ptr;
}


void* cuda_realloc(void* ptr, size_t old_size, size_t new_size) {
    if (ptr == NULL) {
        return cuda_malloc(new_size);
    }

    if (new_size == 0) {
        cudaFreeHost(ptr);
        return NULL;
    }

    void* new_ptr = cuda_malloc(new_size);
    if (new_ptr == NULL) {
        return NULL;
    }

    size_t copy_size = old_size < new_size ? old_size : new_size;
    cudaMemcpy(new_ptr, ptr, copy_size, cudaMemcpyHostToHost);

    cudaFreeHost(ptr);
    return new_ptr;
}

//#define STBI_MALLOC(sz)                     cuda_malloc(sz)
//#define STBI_REALLOC_SIZED(p,oldsz,newsz)   cuda_realloc(p,oldsz,newsz)
//#define STBI_FREE(p)                        cudaFreeHost(p)

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#include <string>
#include <stdio.h>
#include <stdexcept>
#include <regex>

Image::Image(){
    shape = {0,0,0};
    data = nullptr;
}

Image::Image(std::string filename, int channels){
    int dummy;
    shape.z = channels;
    data = (byte*) stbi_load(filename.c_str(), &(shape.x), &(shape.y), &dummy, shape.z);

    if(data == nullptr)
        throw std::runtime_error("Failed to load image " + filename + "': " + stbi_failure_reason());
    
    #ifdef MEM_DEBUG
    printf("Image OPEN   at %p -> %s\n", data, filename.c_str());
    #endif
}

Image& Image::operator=(Image&& other){
    if(this != &other){
        if(data != nullptr) free(data);

        data = other.data;
        shape = other.shape;
        
        other.data = nullptr;
    }

    return *this;
}

void Image::save(std::string filename){
    if(!stbi_write_png(filename.c_str(), shape.x, shape.y, shape.z, data, shape.x * shape.z)){
        throw std::runtime_error("Failed to save image " + filename + "': " + stbi_failure_reason());
    }
}

void Image::save(std::string filename, byte* data, int3 shape){
    if(!stbi_write_png(filename.c_str(), shape.x, shape.y, shape.z, data, shape.x * shape.z)){
        throw std::runtime_error("Failed to save image " + filename + "': " + stbi_failure_reason());
    }
}

Image::~Image(){
    if(data != nullptr){
        #ifdef MEM_DEBUG
        printf("Image DELETE at %p\n", data);
        #endif
        cudaFreeHost(data);
    }
}