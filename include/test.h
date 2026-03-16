#ifndef TEST_H
#define TEST_H

#include <functional>
#include <string>
#include <chrono>
#include <iostream>

typedef std::vector<std::pair<std::string, std::function<void()>>> FuncVector;

#define TEST(func_name) \
    void func_name(); \
    struct func_name##_registrar { \
        func_name##_registrar() { \
            registered_funcs.push_back({#func_name, func_name}); \
        } \
    } func_name##_instance; \
    void func_name()

#define SKIP(func_name) \
    void func_name()

void test();

template<typename F>
void benchmark(const std::string label, F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\033[33mBENCH\033[0m " << label << ": " << elapsed.count() << " ms\n";
}

#endif