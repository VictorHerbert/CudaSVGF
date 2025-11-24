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

TEST(FILTER_LEVEL){
    dim3 blockSize(32,8);
    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);

    test_cache_tile<<<gridSize, blockSize, 30*1024>>>(frame.render);
    cudaDeviceSynchronize();
}

// --------------------------------------------------------------------------------------------------------------

TEST(FILTER){
    dim3 blockSize(32,8);
    dim3 gridSize((shape.x + blockSize.x-1) / blockSize.x, (shape.y + blockSize.y-1) / blockSize.y);
    
    FilterParams params = {
        .type = FilterParams::AVERAGE,
        .depth=1};
    
    int bytecount;
    bytecount = tileByteCount(params, {blockSize.x, blockSize.y});

    filterKernel<<<gridSize, blockSize>>>(frame, params);
    cudaDeviceSynchronize();

    //params.cacheTile = true;
    //bytecount = byteCountTile(params, {blockSize.x, blockSize.y});
    //filterKernel<<<gridSize, blockSize, 20*1024>>>(frame, params);
    //cudaDeviceSynchronize();
}
