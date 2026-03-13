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

const std::string OUTPUT_PATH =  "test/";
const std::string IMAGE_SAMPLE_PATH =  "render/sponza/render/1.png";
const std::string IMAGE_SAMPLE_GBUFFER_PATH =  "render/sponza/$channel$/1.png";
const std::string IMAGE_OUPUT_PATH =  "render/sponza/output/0.png";

FuncVector registered_funcs;

void printTestFuncs(){
    //printf("%d available tests: ", registered_funcs.size());
    for (auto& [name, func] : registered_funcs)
        printf("\t%s\n", name.c_str());
}


#include <chrono>
#include <iostream>

#define BENCHMARK(label, ...) \
do { \
    auto _start = std::chrono::high_resolution_clock::now(); \
    { __VA_ARGS__ } \
    auto _end = std::chrono::high_resolution_clock::now(); \
    auto _elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start); \
    std::cout << label << ": " << _elapsed.count() << " ms\n"; \
} while (0)

template<typename F>
void benchmark(const std::string label, F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << label << ": " << elapsed.count() << " ms\n";
}

void test(std::string wildcard) {
    /*printf("----------------------------------------------------------\n");
    printf("%d available tests: ", registered_funcs.size());
    for (auto& [name, func] : registered_funcs)
        printf("%s ", name.c_str());
    printf("\n----------------------------------------------------------\n");*/

    //printTestFuncs();

    std::regex base_regex(wildcard);

    printf("----------------------------------------------------------\n");
    if (registered_funcs.empty()) {
        printf("No tests found\n");
        printf("----------------------------------------------------------\n");
        return;
    }
    for (auto& [name, func] : registered_funcs) {
        if (!std::regex_match(name, base_regex)) {
            continue;
        }

        try {
            //printf("TEST %s:\n", name.c_str());
            //auto start = std::chrono::high_resolution_clock::now();

            //benchmark("TEST " + name, [&]{
                func();
            //});
            //auto end = std::chrono::high_resolution_clock::now();
            //double frameTime = std::chrono::duration<double, std::milli>(end - start).count();

            //printf("Passed with %.3f ms\n", frameTime);

            printf("\033[33mTEST\033[0m \033[1m%-30s\033[0m \033[32mPASSED\033[0m\n", name.c_str());
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

CpuGBuffer cpuFrame(shape);
CpuVector<uchar4> cpuBuffer(totalSize(shape));

CudaGBuffer cudaFrame(shape);
CudaVector<uchar4> cudaBuffer(totalSize(shape));

FilterParams defaultParams;
dim3 defaultBlockSize(32,8);
dim3 defaultGridSize((shape.x + defaultBlockSize.x-1) / defaultBlockSize.x, (shape.y + defaultBlockSize.y-1) / defaultBlockSize.y);
int2 defaultBlockShape = {defaultBlockSize.x, defaultBlockSize.y};

// --------------------------------------------------------------------------------------------------------------

SKIP(device_stats){
    printGPUProperties();
}

SKIP(image_manipulation){
    Image image3(IMAGE_SAMPLE_PATH, 3);
    image3.save(OUTPUT_PATH + "image_open_save3.png");

    Image image4(IMAGE_SAMPLE_PATH, 4);
    image4.save(OUTPUT_PATH + "image_open_save4.png");
}

SKIP(gbuffer_open){
    CpuGBuffer buffer(IMAGE_SAMPLE_GBUFFER_PATH);
    buffer.renderImg.save(OUTPUT_PATH + "render_out.png");
    buffer.normalImg.save(OUTPUT_PATH + "normal_out.png");
    buffer.albedoImg.save(OUTPUT_PATH + "albedo_out.png");
}

// --------------------------------------------------------------------------------------------------------------

SKIP(atrous_pass_cpu){
    cpuFrame.openImages(IMAGE_SAMPLE_GBUFFER_PATH);

    benchmark(">> BENCH atrous_pass_cpu", [&]{
        atrousFilterPassCpu(cpuFrame.render, cpuFrame.denoised, 0, cpuFrame, defaultParams);
    });

    Image::save(
        IMAGE_OUPUT_PATH,
        (byte*) cpuFrame.denoised,
        {cpuFrame.shape.x, cpuFrame.shape.y, 4}
    );
}

// --------------------------------------------------------------------------------------------------------------

__global__ void atrousFilterPassCudaBaseKernel(GBuffer frame, const FilterParams params, int level){
    atrousFilterPassCudaBase(frame.render, frame.denoised, level, frame, params);
}

SKIP(atrous_pass_cuda_base){
    for(int i = 0; i < 5; i++)
        benchmark(">> BENCH atrous_pass_cuda_base" + std::to_string(i), [&]{
            atrousFilterPassCudaBaseKernel<<<defaultGridSize, defaultBlockSize>>>(cudaFrame, defaultParams, i);
            cudaDeviceSynchronize();
        });
}

// --------------------------------------------------------------------------------------------------------------

SKIP(atrous_cpu){
    cpuFrame.openImages(IMAGE_SAMPLE_GBUFFER_PATH);

    benchmark(">> BENCH atrous_cpu", [&]{
        atrousFilterCpu(cpuFrame, defaultParams);
    });

    Image::save(
        IMAGE_OUPUT_PATH,
        (byte*) cpuFrame.denoised,
        {cpuFrame.shape.x, cpuFrame.shape.y, 4}
    );
}

// --------------------------------------------------------------------------------------------------------------

__global__ void atrousFilterCudaBaseKernel(GBuffer frame, FilterParams params, int level){
    printf("Inside kernel");
    params.depth = level;
    //atrousFilterCudaBase(frame, params);
}

TEST(atrous_cuda_base){
    cpuFrame.openImages(IMAGE_SAMPLE_GBUFFER_PATH);

    int i = 0;
    benchmark(">> BENCH atrous_cuda_base" + std::to_string(i), [&]{
        printf("inside macro\n");
        atrousFilterCudaBaseKernel<<<defaultGridSize, defaultBlockSize>>>(cudaFrame, defaultParams, i);
        cudaDeviceSynchronize();
        printf("finish macro\n");
    });

    Image::save(
        IMAGE_OUPUT_PATH ,
        (byte*) cpuFrame.denoised,
        {cpuFrame.shape.x, cpuFrame.shape.y, 4}
    );
}

// --------------------------------------------------------------------------------------------------------------

SKIP(video_atrous_cpu){
    videoFilterCpu("render/sponza/$channel$/$i$.png", defaultParams);
}

// --------------------------------------------------------------------------------------------------------------

SKIP(video_atrous_gpu){

}