#include "image.h"
#include "utils.h"
#include "extended_math.h"

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

Image::Image(int3 shape){
    this->shape = shape;
    // TODO check if pinned memory needed
    data = (byte*) malloc(totalSize(shape));
}

/*Image::Image(byte* data, int3 shape){
    this->shape = shape;
    // TODO check if deep copy is needed, risc of double free
    this->data = data;
}*/

Image::Image(std::string filename, int channels){
    int dummy;
    shape.z = channels;
    data = (byte*) stbi_load(filename.c_str(), &(shape.x), &(shape.y), &dummy, shape.z);

    if(data == nullptr)
        throw std::runtime_error("Failed to load image " + filename + "': " + stbi_failure_reason());
    
    //printf("Image OPEN   at %p -> %s\n", data, filename.c_str());
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
        //printf("Image DELETE at %p\n", data);
        free(data);
    }
}