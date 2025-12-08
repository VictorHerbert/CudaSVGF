#include "image.h"
#include "filter.cuh"
#include "test.h"
#include "utils.h"

#include "third_party/stb_image.h"

#include <vector>
#include <cassert>
#include <regex>


const std::string OUTPUT_PATH =  "test/";
const std::string IMAGE_SAMPLE_PATH =  "render/sponza/render/1.png";

FuncVector registered_funcs;
auto logStart = std::chrono::high_resolution_clock::now();

void logTime(std::string log){
    auto end = std::chrono::high_resolution_clock::now();
    double frameTime = std::chrono::duration<double, std::milli>(end - logStart).count();

    printf("Reached %s with %.3f ms\n", log.c_str(), frameTime);

    logStart = std::chrono::high_resolution_clock::now();
}

void test(std::string wildcard) {
    printf("----------------------------------------------------------\n");
    printf("%d available tests: ", registered_funcs.size());
    for (auto& [name, func] : registered_funcs)
        printf("%s ", name.c_str());
    printf("\n----------------------------------------------------------\n");

    std::regex base_regex(wildcard);

    for (auto& [name, func] : registered_funcs) {
        if (!std::regex_match(name, base_regex)) {
            continue;
        }

        try {
            printf("TEST %s:\n", name.c_str());
            auto start = std::chrono::high_resolution_clock::now();
            logStart = start;
            func();
            auto end = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(end - start).count();

            printf("Passed with %.3f ms\n", frameTime);
        }
        catch (const std::runtime_error& e) {
            printf("Fail with %s\n", e.what());
        }
        catch (...) {
            printf("Failed\n");
        }
        printf("----------------------------------------------------------\n");
    }
}

// --------------------------------------------------------------------------------------------------------------

int2 shape =  {1920, 1080};
//int2 shape =  {512, 512};
CudaGBuffer frame(shape);

FilterParams defaultParams;
dim3 defaultBlockSize(32,8);
dim3 defaultGridSize((shape.x + defaultBlockSize.x-1) / defaultBlockSize.x, (shape.y + defaultBlockSize.y-1) / defaultBlockSize.y);
int2 defaultBlockShape = {defaultBlockSize.x, defaultBlockSize.y};

// --------------------------------------------------------------------------------------------------------------

SKIP(DEVICE_STATS){
    printGPUProperties();
}

SKIP(IMAGE){
    Image image3(IMAGE_SAMPLE_PATH, 3);
    image3.save(OUTPUT_PATH + "image_open_save3.png");

    Image image4(IMAGE_SAMPLE_PATH, 4);
    image4.save(OUTPUT_PATH + "image_open_save4.png");
}

// --------------------------------------------------------------------------------------------------------------

TEST(TEST_SYNC){
    cudaDeviceSynchronize();
}

KERNEL void test_tileLine(uchar4 *in){
    extern __shared__ uchar4 tile[];
    //for(int i = 0; i < 10; i++)
        tileLine(tile, in, 1000);
}

SKIP(TEST_tileLine){
    dim3 blockSize(32,4);
    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);
    
    // Aligned Tile
    test_tileLine<<<1, blockSize, 4*1000>>>(frame.render);
    cudaDeviceSynchronize();
}

// --------------------------------------------------------------------------------------------------------------

KERNEL void dilatedFilterBaseKernel(GBuffer frame, FilterParams params, int level){
    dilatedFilterBase(frame.render, frame.denoised, level, frame, params);
}

SKIP(LEVEL_FILTER_BASE){
    dim3 blockSize(128,1);
    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

    dilatedFilterBaseKernel<<<gridSize, blockSize>>>(frame, defaultParams, 1);
    cudaDeviceSynchronize();
}

// --------------------------------------------------------------------------------------------------------------

KERNEL void dilatedFilter2Kernel(GBuffer frame, FilterParams params, int level){
    dilatedFilter2(frame.render, frame.denoised, level, frame, params);
}

SKIP(LEVEL_FILTER_2){
    dim3 blockSize(128,1);
    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

    dilatedFilter2Kernel<<<gridSize, blockSize>>>(frame, defaultParams, 1);
    cudaDeviceSynchronize();
}

// --------------------------------------------------------------------------------------------------------------

KERNEL void dilatedFilterLineTileKernel(GBuffer frame, FilterParams params, int level){
    extern __shared__ uchar4 sharedMem[];
    int offset = (blockDim.x + 2*(1<<(level+1)));

    frame.renderTile = sharedMem;
    frame.albedoTile = frame.renderTile + offset;
    frame.normalTile = frame.albedoTile + offset;
    if(blockIdx.x == 0 && blockIdx.y == 0)
        /*if(threadIdx.x == 0 && threadIdx.y == 0){
            printf("buffer0 %p | buffer1 %p\n", frame.buffer[0], frame.buffer[1]);
            printf("render %p | albedo %p | normal %p\n", frame.render, frame.albedo, frame.normal);
        }*/
    dilatedFilterLineTile(frame.render, frame.denoised, level, frame, params);
}

CpuVector<dim3> blocks = {dim3(8,8), dim3(16,16), dim3(32, 1), dim3(32, 4), dim3(32, 16), dim3(32, 32), dim3(64, 16), dim3(128, 8), dim3(256, 4)};

TEST(LEVEL_FILTER_LINE){
    int level = 0;

    for(auto blockSize : {dim3(128)}){
        dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

        int byteCount = 3*4*(blockSize.x + 2*(1<<(level+1)));

        dilatedFilterLineTileKernel<<<gridSize, blockSize, byteCount>>>(frame, defaultParams, level);
        CHECK_CUDA(cudaDeviceSynchronize());

        //std::cout << blockSize.x << " " << blockSize.y << "-> ";
        //logTime("");
    }

}

// --------------------------------------------------------------------------------------------------------------

/*SKIP(FILTER_LEVEL_BENCHMARK){
    CpuVector<dim3> blockSizes = {dim3(8,8), dim3(16, 16), dim3(32,8)};
    CpuVector<int> levels = {0,1,2,3,4};
    CpuVector<FilterParams> params = {
        FilterParams{.tile= FilterParams::AVERAGE},
        FilterParams{.tile= FilterParams::ALBEDO},
        FilterParams{.tile= static_cast<FilterParams::Type>(FilterParams::ALBEDO | FilterParams::NORMAL)}
    };

    for(auto blockSize : blockSizes)
        for(auto level : levels)
            for(auto param : params){
                int bytecount = tileBytes(param, {blockSize.x, blockSize.y});
                dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);
                singledilatedFilterKernel<<<gridSize, blockSize, bytecount>>>(frame, param, level);
                cudaDeviceSynchronize();

                logTime("BLOCK (" + std::to_string(blockSize.x) + "," + std::to_string(blockSize.y) + ")\t LEVEL " + std::to_string(level));
            }
}*/

// --------------------------------------------------------------------------------------------------------------
