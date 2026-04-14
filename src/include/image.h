#ifndef IMAGE_H
#define IMAGE_H

/// @file image.h
/// @brief Image container and I/O utilities.

#include "cuda_utils.h"
#include "vector.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>

typedef unsigned char byte;


/// @brief CPU image container storing raw pixel data.
struct Image {
    int3 shape;
    byte* data;

    /// @brief Default constructor.
    Image();

    /// @brief Constructs an image with allocated storage.
    /// @param shape Image dimensions (width, height, channels or depth).
    Image(int3 shape);

    /// @brief Wraps an external pixel buffer without copying.
    /// @param data Pointer to raw pixel data.
    /// @param shape Image dimensions (width, height, channels or depth).
    Image(byte* data, int3 shape);

    /// @brief Loads an image from disk.
    /// @param filename Path to image file.
    /// @param channels Number of channels to load (default: 4).
    Image(std::string filename, int channels = 4);

    /// @brief Destructor.
    ~Image();

    /// @brief Move assignment operator.
    /// @param other Source image.
    /// @return Reference to this image.
    Image& operator=(Image&& other);

    /// @brief Saves image to disk.
    /// @param filename Output file path.
    void save(std::string filename);

    /// @brief Saves raw image data to disk.
    /// @param filename Output file path.
    /// @param data Raw pixel buffer.
    /// @param shape Image dimensions.
    static void save(std::string filename, byte* data, int3 shape);
};

#endif