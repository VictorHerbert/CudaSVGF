#ifndef TEST_H
#define TEST_H

/// @file test.h
/// @brief Simple testing and benchmarking utilities

#include <functional>
#include <string>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>


/// @brief Entry point for running the full test suite.
/// Typically aggregates multiple CHECK() calls.
void test();

/// @brief Exception type used to skip a test case.
class test_skip : public std::exception {};

/// @brief Skips the current test execution.
/// Throws a test_skip exception which is caught by CHECK().
#define SKIP() throw test_skip()

/// @brief Benchmarks a callable using CPU and CUDA timing.
/// Measures CPU wall time using std::chrono and GPU execution time
/// using CUDA events.
template<typename F>
void BENCH(const std::string& label, F&& func, int it = 1) {
    auto cpu_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    //func();

    cudaEventRecord(start);

    func();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    auto cpu_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

    printf("\033[33mBENCH\033[0m \033[1m%-40s\033[0m\033[0m GPU \033[32m%.2f ms\033[0m\n",
            label.c_str(), gpu_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
}

/// @brief Executes a test case and reports result status.
/// Captures exceptions and classifies execution as PASSED, SKIP, or FAIL.
template<typename F>
void CHECK(const std::string label, F&& func) {
    printf("----------------------------------------------------------\n");
    try {
        func();
        printf("\033[33mTEST \033[0m \033[1m%-40s\033[0m \033[32mPASSED\033[0m\n", label.c_str());
    }
    catch (const test_skip&) {
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