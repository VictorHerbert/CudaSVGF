#include "image.h"
#include "filter.cuh"
#include "test.h"
#include "utils.h"
#include "video.cuh"
#include "metrics.h"
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
    
    auto blockSizes = {dim3(32), dim3(32, 4)};
    auto depths = {1,2,5,10};

    int depth = 5;

//--------------------------------------------------------------------------------------------------------------

    CHECK("cpu_atrous", [&]{
        SKIP();

        //int2 shape = {512, 512};
        CpuGBuffer<uchar3> cpuFrame(shape);
        cpuFrame.openImages("render/sponza/$channel$/1.png");

        BENCH("cpu_atrous", [&]{
            atrousFilterCpu(cpuFrame, 5);
        });

    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("cuda_atrous_base", [&]{
        //SKIP();

        CudaGBuffer<uchar3> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_base3 dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaKernelBase<<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });

    CHECK("cuda_atrous_uchar4", [&]{
        //SKIP();

        CudaGBuffer<uchar4> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_base4 dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaKernelU4<<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });

    CHECK("cuda_atrous_O3", [&]{
        //SKIP();

        CudaGBuffer<uchar4> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_O3 dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaKernelO3<<<gridSize, blockSize>>>(frame.render, frame.denoised, depth,frame);
                cudaDeviceSynchronize();
            });
        }
    });

    CHECK("cuda_atrous_aprox", [&]{
        //SKIP();

        CudaGBuffer<uchar4> frame(shape);

        for(dim3 blockSize : blockSizes){
            dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

            BENCH("cuda_atrous_opt dim " + std::to_string(blockSize.x) + " " + std::to_string(blockSize.y), [&]{
                atrousFilterCudaKernelOpt<<<gridSize, blockSize>>>(frame.render, frame.denoised, depth, frame);
                cudaDeviceSynchronize();
            });
        }
    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("video_cuda", [&]{
        videoFilterCuda(
            "render/sponza/$channel$/$i$.png",
            "render/sponza/output/$i$.png",
            shape,
            5
        );
    });

// --------------------------------------------------------------------------------------------------------------

    CHECK("params_exploration", [&]{
        SKIP();
    });

}