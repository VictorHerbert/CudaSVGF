#include "image.h"
#include "filter.cuh"
#include "test.h"
#include "cuda_utils.h"
#include "animation.cuh"
#include "third_party/stb_image.h"

#include <vector>
#include <cassert>
#include <regex>

#include <iostream>

void test() {

// --------------------------------------------------------------------------------------------------------------

    CHECK("device_stats", [&]{
        printGPUProperties();
    });

// --------------------------------------------------------------------------------------------------------------

    int2 shape = {1920, 1080};

    auto blockSizes = {dim3(32), dim3(32, 2), dim3(32, 4)};
    int depth = 4;

//--------------------------------------------------------------------------------------------------------------

    CHECK("cpu_atrous", [&]{
        //SKIP();

        CpuGBuffer<uchar3> cpuFrame(shape);

        BENCH("cpu_atrous", [&]{
            atrousFilter<CPU, BASE, uchar3>(cpuFrame, 5);
        });

    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("cuda_atrous_u3", [&]{
        //SKIP();
        CudaGBuffer<uchar3> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);
            BENCH("cuda_atrous_u3 dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaPass<BASE><<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                CUDA_KERNEL_ERROR_CHECK();
                cudaDeviceSynchronize();
            });
        }
    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("cuda_atrous_u4", [&]{
        //SKIP();

        CudaGBuffer<uchar4> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_u4 dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaPass<BASE><<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                CUDA_KERNEL_ERROR_CHECK();
                cudaDeviceSynchronize();
            });
        }
    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("cuda_atrous_tile", [&]{
        //SKIP();

        CudaGBuffer<uchar4> frame(shape);

        for(dim3 blockShape : blockSizes){
            dim3 gridShape((frame.shape.x + blockShape.x-1) / blockShape.x, (frame.shape.y + blockShape.y-1) / blockShape.y);

            int2 tileShape = {blockShape.x + 2*ATROUS_RADIUS*(1<<depth), blockShape.y + 2*ATROUS_RADIUS*(1<<depth)};
            int tileSize = totalSize(tileShape);
            int tileBytes = tileSize*sizeof(uchar4);

            assert(tileBytes < 48*1024);

            BENCH("cuda_atrous_tile dim " + std::to_string(blockShape.x) + " " + std::to_string(blockShape.y), [&]{
                atrousFilterCudaPass<TILE><<<gridShape, blockShape, tileBytes>>>(frame.render, frame.denoised, depth, frame);
                CUDA_KERNEL_ERROR_CHECK();
                cudaDeviceSynchronize();
            });
        }
    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("cuda_atrous_aligned", [&]{
        //SKIP();

        CudaGBuffer<uchar4> frame(shape);

        for(dim3 blockShape : blockSizes){
            dim3 gridShape((frame.shape.x + blockShape.x-1) / blockShape.x, (frame.shape.y + blockShape.y-1) / blockShape.y);

            int2 tileShape = {blockShape.x + 2*ATROUS_RADIUS*(1<<depth), blockShape.y + 2*ATROUS_RADIUS*(1<<depth)};
            int tileSize = totalSize(tileShape);
            int tileBytes = tileSize*sizeof(uchar4);

            assert(tileBytes < 48*1024);

            BENCH("cuda_atrous_aligned dim " + std::to_string(blockShape.x) + " " + std::to_string(blockShape.y), [&]{
                atrousFilterCudaPass<ALIGNED><<<gridShape, blockShape, tileBytes>>>(frame.render, frame.denoised, depth, frame);
                CUDA_KERNEL_ERROR_CHECK();
                cudaDeviceSynchronize();
            });
        }
    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("image_io", [&]{
        //SKIP();
        BENCH("image_read_512", [&]{volatile Image img("render/cornell/Render0001.png");});
        Image img_512("render/cornell/Render0001.png");
        BENCH("image_save_512", [&]{img_512.save("build/img_save512.png");});

        BENCH("image_read_hd", [&]{volatile Image img("render/sponza/Render0001.png");});
        Image img_hd("render/sponza/Render0001.png");
        BENCH("image_save_hd", [&]{img_hd.save("build/img_save_hd.png");});
    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("animation_cuda", [&]{
        //SKIP();
        BENCH("animation_cuda_512_1", [&]{animationFilterCuda("render/cornell/", {512, 512}, 14, 5, FilterParams(), 1);});
        BENCH("animation_cuda_512_4", [&]{animationFilterCuda("render/cornell/", {512, 512}, 14, 5, FilterParams(), 4);});
        BENCH("animation_cuda_512_8", [&]{animationFilterCuda("render/cornell/", {512, 512}, 14, 5, FilterParams(), 8);});
        BENCH("animation_cuda_hd_1", [&]{animationFilterCuda("render/sponza/", {1920, 1080}, 14, 5, FilterParams(), 1);});
        BENCH("animation_cuda_hd_4", [&]{animationFilterCuda("render/sponza/", {1920, 1080}, 14, 5, FilterParams(), 4);});
        BENCH("animation_cuda_hd_8", [&]{animationFilterCuda("render/sponza/", {1920, 1080}, 14, 5, FilterParams(), 8);});
    });

}