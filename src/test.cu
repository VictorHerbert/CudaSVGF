#include "image.h"
#include "filter.cuh"
#include "test.h"

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

KERNEL void test_cache_tile(uchar4 *in){
    extern __shared__ uchar4 tile[];
    int2 start = {blockIdx.x * blockDim.x, blockIdx.y * blockDim.y};
    int2 end = {(blockIdx.x+1) * blockDim.x, (blockIdx.y+1) * blockDim.y};
    cacheTile(tile, in, {1920, 1080}, start, end);
}

SKIP(CACHE_TILE){
    dim3 blockSize(32,8);
    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

    test_cache_tile<<<gridSize, blockSize, 30*1024>>>(frame.render);
    cudaDeviceSynchronize();
}


// --------------------------------------------------------------------------------------------------------------

KERNEL void singleLevelFilterKernelBase(GBuffer frame, FilterParams params, int level){
    singleLevelFilterBase(frame.render, frame.denoised, level, frame, params);
}

KERNEL void singleLevelFilterKernel(GBuffer frame, FilterParams params, int level){
    singleLevelFilter(frame.render, frame.denoised, level, frame, params);
}

TEST(FILTER_LEVEL_BASE){
    singleLevelFilterKernelBase<<<defaultGridSize, defaultBlockSize>>>(frame, defaultParams, 0);
    cudaDeviceSynchronize();
}

TEST(FILTER_LEVEL){
    int bytecount = tileBytes(defaultParams, {defaultBlockSize.x, defaultBlockSize.y});
    singleLevelFilterKernel<<<defaultGridSize, defaultBlockSize, bytecount>>>(frame, defaultParams, 0);
    cudaDeviceSynchronize();
}

SKIP(FILTER_LEVEL_BENCHMARK){
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
                singleLevelFilterKernel<<<gridSize, blockSize, bytecount>>>(frame, param, level);
                cudaDeviceSynchronize();

                logTime("BLOCK (" + std::to_string(blockSize.x) + "," + std::to_string(blockSize.y) + ")\t LEVEL " + std::to_string(level));
            }
}
// --------------------------------------------------------------------------------------------------------------

SKIP(FILTER_BASE){
    filterKernelBase<<<defaultGridSize, defaultBlockSize>>>(frame, defaultParams);
    cudaDeviceSynchronize();
}

SKIP(FILTER_TILE_NO_CACHE){
    FilterParams params = {
        .tile=FilterParams::AVERAGE,
        .depth=5
    };

    int bytecount = tileBytes(params, {defaultBlockSize.x, defaultBlockSize.y});

    filterKernel<<<defaultGridSize, defaultBlockSize, bytecount>>>(frame, params);
    cudaDeviceSynchronize();
}

SKIP(FILTER_TILE){
    FilterParams params = {.depth=5};

    int bytecount = tileBytes(params, {defaultBlockSize.x, defaultBlockSize.y});

    filterKernel<<<defaultGridSize, defaultBlockSize, bytecount>>>(frame, params);
    cudaDeviceSynchronize();
}