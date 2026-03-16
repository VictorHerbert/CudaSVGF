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

FuncVector registered_funcs;

void test() {
    printf("----------------------------------------------------------\n");
    for (auto& [name, func] : registered_funcs) {
        try {
            func();
            printf("\033[33mTEST \033[0m \033[1m%-30s\033[0m \033[32mPASSED\033[0m\n", name.c_str());
        }
        catch (const std::runtime_error& e) {
            printf("\033[33mTEST \033[0m \033[1m%-30s\033[0m \033[31mFAIL\033[0m\n", name.c_str());
            printf("%s\n", e.what());
        }
        catch (...) {
            printf("Failed\n");
        }
        printf("----------------------------------------------------------\n");
    }
}

// --------------------------------------------------------------------------------------------------------------

TEST(device_stats){
    printGPUProperties();
}


// --------------------------------------------------------------------------------------------------------------

TEST(filter_cpu){

}

// --------------------------------------------------------------------------------------------------------------

TEST(filter_cuda){

}

// --------------------------------------------------------------------------------------------------------------

TEST(image_cpu){

}

// --------------------------------------------------------------------------------------------------------------

TEST(image_cuda){

}

// --------------------------------------------------------------------------------------------------------------

SKIP(video_cpu){
    benchmark("video_cpu", [&]{
        videoFilterCpu(
            "render/sponza/$channel$/$i$.png",
            "test/cpu/$i$.png",
            {1920, 1080},
            2
        );
    });
}

// --------------------------------------------------------------------------------------------------------------

TEST(video_cuda){
    benchmark("video_cuda", [&]{
        videoFilterCuda(
            "render/sponza/$channel$/$i$.png",
            "test/cuda/$i$.png",
            {1920, 1080},
            5
        );
    });
}