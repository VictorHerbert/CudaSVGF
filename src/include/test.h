#ifndef TEST_H
#define TEST_H

#include <functional>
#include <string>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

void test();

class test_skip : public std::exception {};
#define SKIP() throw test_skip()

template<typename F>
void BENCH(const std::string& label, F&& func, int it = 1) {
    for(int i = 0; i < it; i++){
        auto cpu_start = std::chrono::high_resolution_clock::now();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        func();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        auto cpu_end = std::chrono::high_resolution_clock::now();

        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

        printf("\033[33mBENCH\033[0m it %d \033[1m%-40s\033[0m\033[0m GPU \033[32m%.2f ms\033[0m\n",
            i, label.c_str(), gpu_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

template<typename F>
void CHECK(const std::string label, F&& func) {
    printf("----------------------------------------------------------\n");
    try {
        func();
        printf("\033[33mTEST \033[0m \033[1m%-40s\033[0m \033[32mPASSED\033[0m\n", label.c_str());
    }
    catch (const test_skip& e) {
        printf("\033[33mTEST \033[0m \033[1m%-40s\033[0m \033[30mSKIP\033[0m\n", label.c_str());        
    }
    catch (const std::runtime_error& e) {
        printf("\033[33mTEST \033[0m \033[1m%-40s\033[0m \033[31mFAIL\033[0m\n", label.c_str());
        printf("%s\n", e.what());
    }
    catch (...) {
        printf("Failed\n");
    }
    printf("----------------------------------------------------------\n");

}

#endif