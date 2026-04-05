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
        SKIP();
        printGPUProperties();
    });

// --------------------------------------------------------------------------------------------------------------

    int2 shape = {1920, 1080};
    //int2 shape = {512, 512};

    auto blockSizes = {dim3(32), dim3(32, 2), dim3(32, 4)};
    auto depths = {1,2,5,10};

    int depth = 2;

//--------------------------------------------------------------------------------------------------------------

    CHECK("cpu_atrous", [&]{
        SKIP();

        //int2 shape = {512, 512};
        CpuGFrame<uchar3> cpuFrame(shape);

        BENCH("cpu_atrous", [&]{
            atrousFilterCpu(cpuFrame, 5);
        });

    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("cuda_atrous_no_opt", [&]{
        SKIP();

        CudaGFrame<uchar3> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_no_opt dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaKernelNoOpt<<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });

    CHECK("cuda_atrous_base", [&]{
        SKIP();

        CudaGFrame<uchar3> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_base dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaKernelBase<<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });

    CHECK("cuda_atrous_uchar4", [&]{
        SKIP();

        CudaGFrame<uchar4> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_base4 dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaKernelU4<<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });

    CHECK("cuda_atrous_aprox", [&]{
        //SKIP();

        CudaGFrame<uchar4> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_aprox dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaKernelAprox<<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });

    CHECK("cuda_atrous_tile", [&]{
        //SKIP();

        CudaGFrame<uchar4> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_tile dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                int2 tileShape = {blockSize.x + 2*ATROUS_RADIUS*(1<<depth), blockSize.y + 2*ATROUS_RADIUS*(1<<depth)};
                int tileSize = totalSize(tileShape);
                int tileBytes = tileSize* sizeof(uchar4);

                atrousFilterCudaKernelTile<<<gridSize, blockSize, tileBytes>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });

    CHECK("cuda_atrous_tile_aligned", [&]{
        //SKIP();

        CudaGFrame<uchar4> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_tile_aligned dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                int2 tileShape = {blockSize.x + 2*ATROUS_RADIUS*(1<<depth), blockSize.y + 2*ATROUS_RADIUS*(1<<depth)};
                int tileSize = totalSize(tileShape);
                int tileBytes = tileSize*sizeof(uchar4);

                atrousFilterCudaKernelTileAligned<<<gridSize, blockSize, tileBytes>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });



// --------------------------------------------------------------------------------------------------------------

    CHECK("video_cuda", [&]{
        SKIP();
        videoFilterCuda(
            "render/cornell/", {512, 512}, 10, 5
        );
    });
}